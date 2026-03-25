"""
PyTorch Dataset and DataLoader for CGM time series prediction
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class CGMTimeSeriesDataset(Dataset):
    """
    Dataset for CGM prediction with multimodal inputs.
    Handles sequence creation, normalization, and feature encoding.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int = 144,  # 12 hours
        horizon: int = 12,          # 1 hour
        target_col: str = "cgm_glucose",
        time_varying_known_reals: List[str] = None,
        time_varying_known_cats: List[str] = None,
        time_varying_unknown_reals: List[str] = None,
        static_cols: List[str] = None,
        static_categoricals: List[str] = None,
        static_reals: List[str] = None,
        scalers: Dict = None,
        encoders: Dict = None,
        mode: str = "train",  # train, val, test
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.target_col = target_col
        self.mode = mode

        # Default feature columns
        self.time_varying_known_reals = ["minute_of_day", "hour_of_day",
            "tod_sin", "tod_cos"]
        self.time_varying_known_cats =[]
        self.time_varying_unknown_cats = [ "sleep_stage_state", "activity_name"]
        self.time_varying_unknown_reals = [
            "cgm_glucose", "cgm_lag_1", "cgm_lag_3", "cgm_lag_6", "cgm_diff_1", "cgm_diff_3",
            "cgm_rolling_mean_6", "cgm_rolling_std_6","heart_rate",
            "respiratory_rate", "stress_level", "movement"]

        # Static features
        self.static_categoricals = [
            "participant_id", "clinical_site"
        ]
        self.static_reals = ["age", "BMI", "wth"]
        self.static_cols = (self.static_categoricals + self.static_reals)

        # Store data
        self.data = data.copy()
        self.participant_ids = data["participant_id"].unique()

        # Initialize or use provided scalers/encoders
        self.scalers = scalers or {}
        self.encoders = encoders or {}

        if mode == "train":
            self._fit_scalers_encoders()

        # Apply transformations
        self._transform_data()

        # Create index mapping for efficient sampling
        self._create_index_map()

    def _fit_scalers_encoders(self):
        """Fit scalers and encoders on training data"""
        # Fit scalers for real-valued columns (time-varying + static)
        all_reals = self.time_varying_known_reals + self.time_varying_unknown_reals + self.static_reals

        for col in all_reals:
            if col in self.data.columns:
                scaler = StandardScaler()
                valid_data = self.data[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    scaler.fit(valid_data)
                    self.scalers[col] = scaler

        # Fit encoders for categorical columns (time-varying + static)
        for col in self.time_varying_unknown_cats + self.static_categoricals:
            if col in self.data.columns:
                encoder = LabelEncoder()
                vals = self.data[col].astype(str).fillna("unknown").tolist()
                encoder.fit(vals + ["unknown"])  # ensure "unknown" is always a known class
                self.encoders[col] = encoder

    def _transform_data(self):
        """Apply scalers and encoders to data"""
        # Scale real-valued columns
        for col, scaler in self.scalers.items():
            if col in self.data.columns:
                mask = self.data[col].notna()
                self.data.loc[mask, f"{col}_scaled"] = scaler.transform(
                    self.data.loc[mask, col].values.reshape(-1, 1)
                ).flatten()

        # Encode categorical columns
        for col, encoder in self.encoders.items():
            if col in self.data.columns:
                known = set(encoder.classes_)
                vals = self.data[col].astype(str).fillna("unknown")
                vals = vals.where(vals.isin(known), other="unknown")
                self.data[f"{col}_encoded"] = encoder.transform(vals)

    def _create_index_map(self):
        """Create mapping from global index to (participant, local_idx)"""
        self.index_map = []
        self.participant_data = {}

        for pid in self.participant_ids:
            pdf = self.data[self.data["participant_id"] == pid].reset_index(drop=True)
            self.participant_data[pid] = pdf

            # Create valid starting positions (need context + horizon)
            max_start = len(pdf) - self.context_length - self.horizon
            if max_start > 0:
                for local_idx in range(max_start):
                    self.index_map.append((pid, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, local_idx = self.index_map[idx]
        pdf = self.participant_data[pid]

        # Extract sequences
        encoder_start = local_idx
        encoder_end = local_idx + self.context_length
        decoder_end = encoder_end + self.horizon

        encoder_data = pdf.iloc[encoder_start:encoder_end]
        decoder_data = pdf.iloc[encoder_end:decoder_end]

        # Prepare encoder continuous features (known + unknown reals)
        encoder_cont_cols = [f"{c}_scaled" for c in self.time_varying_known_reals if f"{c}_scaled" in pdf.columns]
        encoder_cont_cols += [f"{c}_scaled" for c in self.time_varying_unknown_reals if f"{c}_scaled" in pdf.columns]

        encoder_cont = encoder_data[encoder_cont_cols].fillna(0).values.astype(np.float32)

        # Prepare decoder continuous features (known reals + sensor data available during prediction)
        # Decoder includes time features + real-time sensor measurements (if available)
        decoder_cont_cols = [f"{c}_scaled" for c in self.time_varying_known_reals if f"{c}_scaled" in pdf.columns]
        decoder_cont = decoder_data[decoder_cont_cols].fillna(0).values.astype(np.float32)

        # Prepare encoder categorical features
        encoder_cat_cols = [f"{c}_encoded" for c in self.time_varying_unknown_cats if f"{c}_encoded" in pdf.columns]
        encoder_cat = encoder_data[encoder_cat_cols].fillna(0).values.astype(np.int64)

        # Prepare decoder categorical features (known cats only — unknown not available at forecast time)
        decoder_known_cat_cols = [f"{c}_encoded" for c in self.time_varying_known_cats if f"{c}_encoded" in pdf.columns]
        decoder_cat = decoder_data[decoder_known_cat_cols].fillna(0).values.astype(np.int64) if decoder_known_cat_cols else np.zeros((self.horizon, 0), dtype=np.int64)

        # Target (future CGM values)
        target = decoder_data[self.target_col].values.astype(np.float32)

        # Target scale (for denormalization)
        target_mean = encoder_data[self.target_col].mean()
        target_std = encoder_data[self.target_col].std()
        if np.isnan(target_std) or target_std == 0:
            target_std = 1.0

        # Static features
        # Categorical static features
        static_cat_cols = [f"{c}_encoded" for c in self.static_categoricals if f"{c}_encoded" in pdf.columns]
        static_cat = pdf[static_cat_cols].iloc[0].values.astype(np.int64) if static_cat_cols else np.array([], dtype=np.int64)

        # Real-valued static features (scaled)
        static_real_cols = [f"{c}_scaled" for c in self.static_reals if f"{c}_scaled" in pdf.columns]
        static_real = pdf[static_real_cols].iloc[0].fillna(0).values.astype(np.float32) if static_real_cols else np.array([], dtype=np.float32)

        # Lengths
        encoder_length = self.context_length
        decoder_length = self.horizon

        # Combine static features with encoder/decoder sequences for TFT compatibility
        # TFT expects static features to be included in the categorical arrays

        # For encoder: prepend static categoricals to each timestep
        encoder_cat_with_static = torch.cat([
            torch.tensor(static_cat, dtype=torch.long).unsqueeze(0).expand(encoder_length, -1),
            torch.tensor(encoder_cat, dtype=torch.long)
        ], dim=-1) if static_cat.size > 0 else torch.tensor(encoder_cat, dtype=torch.long)

        # For decoder: prepend static categoricals to each timestep
        decoder_cat_with_static = torch.cat([
            torch.tensor(static_cat, dtype=torch.long).unsqueeze(0).expand(decoder_length, -1),
            torch.tensor(decoder_cat, dtype=torch.long)
        ], dim=-1) if static_cat.size > 0 else torch.tensor(decoder_cat, dtype=torch.long)

        # Same for static reals
        encoder_cont_with_static = torch.cat([
            torch.tensor(static_real, dtype=torch.float32).unsqueeze(0).expand(encoder_length, -1),
            torch.tensor(encoder_cont, dtype=torch.float32)
        ], dim=-1) if static_real.size > 0 else torch.tensor(encoder_cont, dtype=torch.float32)

        decoder_cont_with_static = torch.cat([
            torch.tensor(static_real, dtype=torch.float32).unsqueeze(0).expand(decoder_length, -1),
            torch.tensor(decoder_cont, dtype=torch.float32)
        ], dim=-1) if static_real.size > 0 else torch.tensor(decoder_cont, dtype=torch.float32)

        return {
            "encoder_cont": encoder_cont_with_static,
            "encoder_cat": encoder_cat_with_static,
            "decoder_cont": decoder_cont_with_static,
            "decoder_cat": decoder_cat_with_static,
            "target": torch.tensor(target, dtype=torch.float32).unsqueeze(-1),  # TFT expects (L, 1)
            "target_scale": torch.tensor([target_mean, target_std], dtype=torch.float32),
            "encoder_lengths": torch.tensor(encoder_length, dtype=torch.long),
            "decoder_lengths": torch.tensor(decoder_length, dtype=torch.long),
            # Keep original for reference if needed
            "static_cat": torch.tensor(static_cat, dtype=torch.long),
            "static_real": torch.tensor(static_real, dtype=torch.float32),
        }

    def get_scalers_encoders(self) -> Tuple[Dict, Dict]:
        """Return scalers and encoders for use in validation/test sets"""
        return self.scalers, self.encoders

    def save_scalers_encoders(self, path: Path):
        """Save scalers and encoders to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "scalers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)
        with open(path / "encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)

    @staticmethod
    def load_scalers_encoders(path: Path) -> Tuple[Dict, Dict]:
        """Load scalers and encoders from disk"""
        path = Path(path)
        with open(path / "scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        with open(path / "encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return scalers, encoders


def create_dataloaders(
    data: pd.DataFrame,
    context_length: int = 144,
    horizon: int = 12,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    target_col: str = "cgm_glucose",
    time_varying_known_reals: List[str] = None,
    time_varying_known_categoricals: List[str] = None,
    time_varying_unknown_reals: List[str] = None,
    static_categoricals: List[str] = None,
    static_reals: List[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict]:
    """
    Create train/val/test dataloaders with temporal split per participant.

    Each participant's sequence is split temporally:
    - Last 1 hour (12 samples): test set
    - Second-to-last 1 hour (12 samples): validation set
    - Remaining earlier data: training set

    For validation: input is all train data, predict the val hour
    For test: input is train+val data, predict the test hour
    """
    participants = data["participant_id"].unique()

    train_data_list = []
    val_data_list = []
    test_data_list = []

    # CGM data is sampled at 5-minute intervals
    # 1 hour = 12 samples (60 min / 5 min)
    samples_per_hour = 12
    val_test_buffer = 2 * samples_per_hour  # 2 hours for val + test

    print(f"Processing {len(participants)} participants with temporal splits...")

    for pid in participants:
        participant_data = data[data["participant_id"] == pid].sort_values("timestamp").reset_index(drop=True)
        n_samples = len(participant_data)

        # Need at least context_length + val + test samples
        min_samples = context_length + val_test_buffer

        if n_samples < min_samples:
            print(f"Skipping participant {pid}: only {n_samples} samples (need {min_samples})")
            continue

        # Split indices
        test_start = n_samples - samples_per_hour  # Last 1 hour for test
        val_start = test_start - samples_per_hour   # Second-to-last 1 hour for val
        train_end = val_start                        # Everything before val

        # Train: all data up to validation period
        train_data_list.append(participant_data.iloc[:train_end])

        # Val: input is train data, output is val period
        # We include the val period in the dataset for creating sequences
        val_data_list.append(participant_data.iloc[:test_start])

        # Test: input is train+val, output is test period
        # We include the test period in the dataset for creating sequences
        test_data_list.append(participant_data)

    train_data = pd.concat(train_data_list, ignore_index=True)
    val_data = pd.concat(val_data_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Participants included: {len(train_data_list)}")

    # Create datasets (pass all feature config from config.py)
    dataset_kwargs = dict(
        target_col=target_col,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_cats=time_varying_known_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
    )

    train_dataset = CGMTimeSeriesDataset(
        train_data, context_length, horizon, mode="train", **dataset_kwargs
    )
    scalers, encoders = train_dataset.get_scalers_encoders()

    val_dataset = CGMTimeSeriesDataset(
        val_data, context_length, horizon,
        scalers=scalers, encoders=encoders, mode="val", **dataset_kwargs
    )
    test_dataset = CGMTimeSeriesDataset(
        test_data, context_length, horizon,
        scalers=scalers, encoders=encoders, mode="test", **dataset_kwargs
    )

    # ── Dataset feature summary ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"Dataset Feature Configuration")
    print(f"{'='*60}")
    print(f"  Target column: {train_dataset.target_col}")
    print(f"  Context length: {context_length} steps ({context_length * 5 / 60:.1f} hours)")
    print(f"  Prediction horizon: {horizon} steps ({horizon * 5 / 60:.1f} hours)")

    print(f"\n  Static Categorical Features ({len(train_dataset.static_categoricals)}):")
    for col in train_dataset.static_categoricals:
        if col in train_dataset.encoders:
            n_classes = len(train_dataset.encoders[col].classes_)
            print(f"    - {col}: {n_classes} unique values")
        else:
            print(f"    - {col} (not in data)")

    print(f"\n  Static Real Features ({len(train_dataset.static_reals)}):")
    for col in train_dataset.static_reals:
        scaled = col in train_dataset.scalers
        print(f"    - {col} {'(scaled)' if scaled else '(not in data)'}")

    print(f"\n  Time-Varying Known Reals ({len(train_dataset.time_varying_known_reals)}):")
    for col in train_dataset.time_varying_known_reals:
        print(f"    - {col}")

    print(f"\n  Time-Varying Unknown Categoricals ({len(train_dataset.time_varying_unknown_cats)}):")
    for col in train_dataset.time_varying_unknown_cats:
        if col in train_dataset.encoders:
            n_classes = len(train_dataset.encoders[col].classes_)
            print(f"    - {col}: {n_classes} unique values (in encoder + decoder)")
        else:
            print(f"    - {col} (not in data)")

    print(f"\n  Time-Varying Unknown Reals ({len(train_dataset.time_varying_unknown_reals)}):")
    for col in train_dataset.time_varying_unknown_reals:
        print(f"    - {col}")


    # Dimension summary from a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\n  Encoder dimensions:")
        print(f"    Continuous: {sample['encoder_cont'].shape}  "
              f"({len(train_dataset.static_reals)} static_reals + "
              f"{len(train_dataset.time_varying_known_reals)} known_reals + "
              f"{len(train_dataset.time_varying_unknown_reals)} unknown_reals)")
        print(f"    Categorical: {sample['encoder_cat'].shape}  "
              f"({len(train_dataset.static_categoricals)} static_cats + "
              f"{len(train_dataset.time_varying_unknown_cats)} unknown_cats)")
        print(f"\n  Decoder dimensions:")
        print(f"    Continuous: {sample['decoder_cont'].shape}  "
              f"({len(train_dataset.static_reals)} static_reals + "
              f"{len(train_dataset.time_varying_known_reals)} known_reals)")
        print(f"    Categorical: {sample['decoder_cat'].shape}  "
              f"({len(train_dataset.static_categoricals)} static_cats + "
              f"{len(train_dataset.time_varying_known_cats)} known_cats)")
        print(f"    Target: {sample['target'].shape}")
    print(f"{'='*60}")

    # Create dataloaders
    # Disable pin_memory when num_workers=0 to save memory
    use_pin_memory = num_workers > 0

    # shuffle=False: Lightning's DDP adds DistributedSampler which handles shuffling
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create feature_info dict for model initialization
    feature_info = {
        "static_categoricals": train_dataset.static_categoricals,
        "static_reals": train_dataset.static_reals,
        "time_varying_known_categoricals": train_dataset.time_varying_known_cats,
        "time_varying_known_reals": train_dataset.time_varying_known_reals,
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": train_dataset.time_varying_unknown_reals,
        "target_col": train_dataset.target_col,
    }

    return train_loader, val_loader, test_loader, train_dataset, feature_info
