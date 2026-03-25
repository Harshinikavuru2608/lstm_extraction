"""
Extract LSTM latent embeddings from a trained LSTMForecastModel.

Three extraction modes (--mode):

  0  LAST-STEP (one row per window):
       For each stride-1 window, record the LSTM hidden state at the last
       encoder position.  Loses the first (context_length-1) timesteps per
       participant because those never appear as the "last step" of any window.

  1  ALL-STEPS (context_length rows per window):
       Record hidden states at EVERY encoder position in every window.
       Captures everything but produces massive output (~288× larger than mode 0)
       and contains heavy redundancy: each raw timestamp appears in up to 288
       different windows with different amounts of context.

  2  UNIQUE-TIMESTEPS (one row per timestamp, recommended):
       For each (participant, timestamp), use the longest available context:
         - timestamps 0 … context_length-2: left-zero-padded shorter windows
         - timestamps context_length-1 … N-horizon-1: full context_length windows
       No data loss.  No redundancy.  Same output size as the original data.
       Adds 'context_steps_used' column (= min(t+1, context_length)).

Output columns (all modes):
  participant_id, timestamp / window_end_timestamp,
  study_group, clinical_site, age, BMI, wth,
  minute_of_day, hour_of_day, tod_sin, tod_cos,
  heart_rate, respiratory_rate, stress_level, activity_level, movement,
  sleep_stage, sleep_stage_state, activity_name,
  cgm_glucose, cgm_lag_1/3/6, cgm_diff_1/3, cgm_rolling_mean/std_6,
  [context_steps_used  — mode 2 only]
  latent_dim_0 … latent_dim_{hidden_size-1}

Usage:
  python extract_latents_lstm.py \\
    --model_pt   logs/lstm_L288_H24/final_model.pt \\
    --processed_parquet /path/to/processed_data_3.0.parquet \\
    --out_path   logs/lstm_L288_H24/latents_lstm_288_24.parquet \\
    --batch_size 256 \\
    --device     cuda \\
    --mode       2
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import torch
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import CGMTimeSeriesDataset
from models.lstm_model import LSTMForecastModel
from configs.config import DataConfig


# ──────────────────────────────────────────────────────────────
# Raw feature columns to pull from the original (unscaled) data
# ──────────────────────────────────────────────────────────────
RAW_FEATURE_COLS = [
    # Identifiers / metadata
    "participant_id", "timestamp",
    "study_group", "clinical_site",
    # Static real features
    "age", "BMI", "wth",
    # Time features
    "minute_of_day", "hour_of_day", "tod_sin", "tod_cos",
    # Sensor / wearable signals
    "heart_rate", "respiratory_rate", "stress_level",
    "activity_level", "movement",
    # Sleep / activity categories
    "sleep_stage", "sleep_stage_state", "activity_name",
    # CGM features (target + variability)
    "cgm_glucose",
    "cgm_lag_1", "cgm_lag_3", "cgm_lag_6",
    "cgm_diff_1", "cgm_diff_3",
    "cgm_rolling_mean_6", "cgm_rolling_std_6",
]


# ──────────────────────────────────────────────────────────────
# Dataset reconstruction
# ──────────────────────────────────────────────────────────────

def _build_datasets(data, data_cfg, context_length, horizon):
    """
    Fit scalers/encoders on the training split (same split used during training),
    then build an 'all' dataset covering every valid window in the full data.
    """
    samples_per_hour  = 12
    val_test_buffer   = 2 * samples_per_hour

    train_list = []
    for pid in data["participant_id"].unique():
        pdata = (data[data["participant_id"] == pid]
                 .sort_values("timestamp")
                 .reset_index(drop=True))
        n = len(pdata)
        if n < context_length + val_test_buffer:
            continue
        test_start = n - samples_per_hour
        val_start  = test_start - samples_per_hour
        train_list.append(pdata.iloc[:val_start])

    train_data = pd.concat(train_list, ignore_index=True)
    print(f"  Train participants : {train_data['participant_id'].nunique()}, "
          f"rows: {len(train_data):,}")

    dataset_kwargs = dict(
        target_col=data_cfg.cgm_col,
        time_varying_known_reals=data_cfg.time_varying_known_reals,
        time_varying_known_cats=data_cfg.time_varying_known_categoricals,
        time_varying_unknown_reals=data_cfg.time_varying_unknown_reals,
        static_categoricals=data_cfg.static_categoricals,
        static_reals=data_cfg.static_reals,
    )

    train_ds = CGMTimeSeriesDataset(
        train_data, context_length, horizon, mode="train", **dataset_kwargs
    )
    scalers, encoders = train_ds.get_scalers_encoders()

    # 'all' dataset: full data with train scalers (mode 0 and 1 use this)
    all_ds = CGMTimeSeriesDataset(
        data, context_length, horizon,
        scalers=scalers, encoders=encoders, mode="val", **dataset_kwargs
    )

    return train_ds, all_ds, scalers, encoders


# ──────────────────────────────────────────────────────────────
# Raw-value helpers
# ──────────────────────────────────────────────────────────────

def _raw_row_to_dict(raw_row, pdf):
    """Extract RAW_FEATURE_COLS from a pandas Series (one row of participant_data)."""
    row = {}
    for col in RAW_FEATURE_COLS:
        if col not in pdf.columns:
            row[col] = None
            continue
        val = raw_row[col]
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val)
        row[col] = val
    return row


# ──────────────────────────────────────────────────────────────
# Mode 2 helper: build scaled encoder_cont matrix for a participant
# ──────────────────────────────────────────────────────────────

def _build_encoder_cont_matrix(pid, all_ds):
    """
    Reconstruct the encoder_cont feature matrix for participant `pid`
    in the exact same column order as CGMTimeSeriesDataset.__getitem__:
        [static_reals_scaled (3), known_reals_scaled (4), unknown_reals_scaled (12)]
        = 19 features total

    Returns np.array of shape (N, input_size), float32.
    """
    pdf = all_ds.participant_data[pid]
    N   = len(pdf)

    # Static reals (same for every row)
    static_real_cols = [
        f"{c}_scaled" for c in all_ds.static_reals
        if f"{c}_scaled" in pdf.columns
    ]
    static_real  = pdf[static_real_cols].iloc[0].fillna(0).values.astype(np.float32)
    static_block = np.tile(static_real, (N, 1))  # (N, S)

    # Time-varying: known reals then unknown reals (same order as __getitem__)
    known_cols   = [f"{c}_scaled" for c in all_ds.time_varying_known_reals
                    if f"{c}_scaled" in pdf.columns]
    unknown_cols = [f"{c}_scaled" for c in all_ds.time_varying_unknown_reals
                    if f"{c}_scaled" in pdf.columns]
    tv_matrix    = pdf[known_cols + unknown_cols].fillna(0).values.astype(np.float32)

    feat_matrix  = np.concatenate([static_block, tv_matrix], axis=1)  # (N, F)
    return feat_matrix, pdf


# ──────────────────────────────────────────────────────────────
# Parquet incremental writer helper
# ──────────────────────────────────────────────────────────────

def _write_chunk(chunk_data, pq_writer_ref, out_path):
    """Convert chunk_data (list[dict]) to parquet, initialise writer on first call."""
    chunk_df = pd.DataFrame(chunk_data)
    table    = pa.Table.from_pandas(chunk_df, preserve_index=False)
    if pq_writer_ref[0] is None:
        pq_writer_ref[0] = pq.ParquetWriter(str(out_path), table.schema)
    pq_writer_ref[0].write_table(table)
    return len(chunk_df)


# ──────────────────────────────────────────────────────────────
# Mode 0/1: extraction via DataLoader (stride-1 windows)
# ──────────────────────────────────────────────────────────────

def _extract_modes_0_1(model, all_ds, device, batch_size, mode, out_path):
    loader = DataLoader(
        all_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    context_length = all_ds.context_length
    pq_writer_ref  = [None]
    total_windows  = 0
    total_rows     = 0
    sample_idx     = 0
    H              = None

    print(f"  Batches : {len(loader)}  |  Windows : {len(all_ds):,}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            B_actual = batch["encoder_cont"].shape[0]

            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # lstm_out: (B, L_enc, hidden_size)
            _, lstm_out = model(batch_dev, return_embeddings=True)
            lstm_np = lstm_out.cpu().numpy()
            H = lstm_np.shape[2]

            global_indices = range(sample_idx, sample_idx + B_actual)

            chunk_data = []

            if mode == 0:
                # ── Last encoder timestep only ───────────────
                latent = lstm_np[:, -1, :]  # (B, H)
                for i, idx in enumerate(global_indices):
                    pid, local_idx = all_ds.index_map[idx]
                    pdf      = all_ds.participant_data[pid]
                    last_pos = local_idx + context_length - 1
                    raw_row  = pdf.iloc[last_pos]

                    row = _raw_row_to_dict(raw_row, pdf)
                    # rename timestamp → window_end_timestamp
                    if "timestamp" in row:
                        row["window_end_timestamp"] = row.pop("timestamp")
                    for d in range(H):
                        row[f"latent_dim_{d}"] = float(latent[i, d])
                    chunk_data.append(row)

            else:  # mode == 1
                # ── Every encoder timestep ───────────────────
                for i, idx in enumerate(global_indices):
                    pid, local_idx = all_ds.index_map[idx]
                    pdf        = all_ds.participant_data[pid]
                    last_ts    = pdf.iloc[local_idx + context_length - 1].get("timestamp", None)

                    for t in range(context_length):
                        raw_row = pdf.iloc[local_idx + t]
                        row = _raw_row_to_dict(raw_row, pdf)
                        # Keep 'timestamp' as raw_timestamp; add window identifier
                        if "timestamp" in row:
                            row["raw_timestamp"]        = row.pop("timestamp")
                        row["window_end_timestamp"] = last_ts
                        row["timestep_in_window"]   = t
                        for d in range(H):
                            row[f"latent_dim_{d}"] = float(lstm_np[i, t, d])
                        chunk_data.append(row)

            total_rows    += _write_chunk(chunk_data, pq_writer_ref, out_path)
            total_windows += B_actual
            sample_idx    += B_actual

            if batch_idx < 2:
                r = chunk_data[0]
                dims = ", ".join(f"{r[f'latent_dim_{d}']:.4f}" for d in range(min(5, H)))
                pid_val = r.get("participant_id", "?")
                cgm_val = r.get("cgm_glucose", float("nan"))
                print(f"  [smoke batch {batch_idx}] pid={pid_val}  "
                      f"cgm={cgm_val:.2f}  latent[:5]=[{dims}]")

            if (batch_idx + 1) % 50 == 0:
                print(f"  batch {batch_idx+1}/{len(loader)} — "
                      f"{total_windows:,} windows / {total_rows:,} rows")

    if pq_writer_ref[0]:
        pq_writer_ref[0].close()

    return total_windows, total_rows, H


# ──────────────────────────────────────────────────────────────
# Mode 2: unique timesteps, left-zero-padded
# ──────────────────────────────────────────────────────────────

def _extract_mode2(model, all_ds, device, batch_size, out_path):
    """
    For every (participant, timestep t) where t in [0, N-horizon-1]:
      - If t < context_length: build a context_length-step window that is
        zero-padded on the left for (context_length - t - 1) steps, then
        actual data from rows [0 … t].
      - If t >= context_length: standard window [t-context_length+1 … t].

    Extract LSTM hidden state at position context_length-1 (last step).
    Records 'context_steps_used' = min(t+1, context_length).
    """
    context_length  = all_ds.context_length
    horizon         = all_ds.horizon
    pq_writer_ref   = [None]
    total_rows      = 0
    total_pids      = 0
    H               = None

    all_pids = list(all_ds.participant_data.keys())
    print(f"  Participants to process: {len(all_pids)}")

    for pid_idx, pid in enumerate(all_pids):
        feat_matrix, pdf = _build_encoder_cont_matrix(pid, all_ds)
        N        = len(pdf)
        n_valid  = N - horizon   # timesteps 0 … N-horizon-1
        if n_valid <= 0:
            continue

        # Left-pad feature matrix with (context_length-1) zero rows
        pad     = np.zeros((context_length - 1, feat_matrix.shape[1]), dtype=np.float32)
        padded  = np.concatenate([pad, feat_matrix], axis=0)  # (L-1+N, F)

        chunk_data = []

        for batch_start in range(0, n_valid, batch_size):
            batch_end = min(batch_start + batch_size, n_valid)
            B = batch_end - batch_start

            # For timestep t, window in padded array = rows [t : t+context_length]
            windows = np.stack(
                [padded[t: t + context_length] for t in range(batch_start, batch_end)],
                axis=0,
            )  # (B, context_length, F)

            batch_tensor = {
                "encoder_cont": torch.from_numpy(windows).to(device)
            }

            with torch.no_grad():
                _, lstm_out = model(batch_tensor, return_embeddings=True)
            latent = lstm_out[:, -1, :].cpu().numpy()  # (B, H)
            H = latent.shape[1]

            for i, t in enumerate(range(batch_start, batch_end)):
                raw_row = pdf.iloc[t]
                row = _raw_row_to_dict(raw_row, pdf)
                row["context_steps_used"] = int(min(t + 1, context_length))
                for d in range(H):
                    row[f"latent_dim_{d}"] = float(latent[i, d])
                chunk_data.append(row)

        total_rows += _write_chunk(chunk_data, pq_writer_ref, out_path)
        total_pids += 1

        if pid_idx < 2:
            r = chunk_data[0]
            dims = ", ".join(f"{r[f'latent_dim_{d}']:.4f}" for d in range(min(5, H)))
            print(f"  [smoke pid {pid_idx}] pid={pid}  N={N}  "
                  f"valid_steps={n_valid}  "
                  f"context[0]={chunk_data[0]['context_steps_used']}  "
                  f"context[-1]={chunk_data[-1]['context_steps_used']}  "
                  f"latent[:5]=[{dims}]")

        if (pid_idx + 1) % 200 == 0:
            print(f"  [{pid_idx+1}/{len(all_pids)} participants]  "
                  f"{total_rows:,} rows written")

    if pq_writer_ref[0]:
        pq_writer_ref[0].close()

    return total_pids, total_rows, H


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def extract(args):
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    print(f"Device: {device}")

    # ── Load hyperparameters ─────────────────────────────────
    hparam_path = Path(args.model_pt).parent / "hyperparameters.json"
    with open(hparam_path) as f:
        hparams = json.load(f)

    context_length = hparams["context_length"]
    horizon        = hparams["horizon"]
    hidden_size    = hparams["hidden_size"]
    num_layers     = hparams["num_layers"]
    input_size     = hparams["input_size"]
    dropout        = hparams["dropout"]
    learning_rate  = hparams["learning_rate"]

    print(f"Hyperparameters: context={context_length}, horizon={horizon}, "
          f"hidden={hidden_size}, layers={num_layers}, input_features={input_size}")

    # ── Load data ────────────────────────────────────────────
    data_cfg = DataConfig()
    print(f"\nLoading data from {args.processed_parquet}")
    data = pd.read_parquet(args.processed_parquet)
    print(f"  Rows: {len(data):,}  |  Participants: {data['participant_id'].nunique()}")

    present = [c for c in RAW_FEATURE_COLS if c in data.columns]
    missing = [c for c in RAW_FEATURE_COLS if c not in data.columns]
    print(f"  Raw feature cols found  : {len(present)}")
    if missing:
        print(f"  Raw feature cols missing: {missing}")

    # ── Build datasets ───────────────────────────────────────
    print("\nBuilding datasets (fit scalers on train split) ...")
    train_ds, all_ds, scalers, encoders = _build_datasets(
        data, data_cfg, context_length, horizon
    )
    print(f"  All-split windows : {len(all_ds):,}")

    # ── Load model ───────────────────────────────────────────
    print(f"\nLoading model from {args.model_pt}")
    ckpt = torch.load(args.model_pt, map_location="cpu", weights_only=False)
    model = LSTMForecastModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Latent dims: {hidden_size}")

    # ── Setup output ─────────────────────────────────────────
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_desc = {
        0: "last encoder step only  (1 row/window, loses first "
           f"{context_length-1} timesteps/participant)",
        1: f"all encoder steps  ({context_length} rows/window, heavy redundancy)",
        2: "unique timesteps  (1 row/timestamp, zero-padded early steps — RECOMMENDED)",
    }
    print(f"\nExtraction mode {args.mode}: {mode_desc[args.mode]}")
    print("-" * 60)

    # ── Run extraction ───────────────────────────────────────
    if args.mode in (0, 1):
        total_windows, total_rows, H = _extract_modes_0_1(
            model, all_ds, device, args.batch_size, args.mode, out_path
        )
        print(f"\nDone!  Windows: {total_windows:,}  |  Rows: {total_rows:,}")
    else:
        total_pids, total_rows, H = _extract_mode2(
            model, all_ds, device, args.batch_size, out_path
        )
        print(f"\nDone!  Participants: {total_pids:,}  |  Rows: {total_rows:,}")

    print(f"  Latent dims : {H}")
    print(f"  Output      : {out_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract LSTM latent embeddings from trained LSTMForecastModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_pt", type=str, required=True,
        help="Path to final_model.pt saved by train_lstm.py",
    )
    parser.add_argument(
        "--processed_parquet", type=str, required=True,
        help="Path to preprocessed parquet (same file used during training)",
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Inference batch size (default: 256)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--mode", type=int, default=2, choices=[0, 1, 2],
        help=(
            "0: last encoder step per window (fast, loses first L-1 timesteps). "
            "1: every encoder step per window (huge output, redundant). "
            "2: one unique row per timestamp, zero-padded early steps (recommended)."
        ),
    )
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
