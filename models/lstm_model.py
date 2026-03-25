"""
Stacked LSTM forecasting model for CGM prediction.

Architecture:
  encoder_cont (B, L_enc, F_enc)
    → linear projection → (B, L_enc, hidden_size)
    → stacked LSTM      → (B, L_enc, hidden_size)
    → last `horizon` steps
    → linear head       → (B, horizon)

The model consumes the same batch dict produced by data/dataset.py.
Only encoder_cont is used as input; target is the raw future CGM sequence.
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Dict, Optional, Tuple


class LSTMForecastModel(pl.LightningModule):
    """Stacked LSTM encoder with a direct-horizon linear forecast head."""

    def __init__(
        self,
        input_size: int,          # encoder_cont feature dim (F_enc)
        hidden_size: int = 128,
        num_layers: int = 4,
        horizon: int = 12,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.horizon = horizon
        self.learning_rate = learning_rate

        # Project raw features to hidden_size before the LSTM
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Stacked LSTM (dropout only applied between layers, not on the last)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        # Project last hidden state to all horizon steps at once
        self.head = nn.Linear(hidden_size, horizon)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ):
        """
        Args:
            batch: dict with key "encoder_cont" of shape (B, L_enc, F_enc)
            return_embeddings: if True, also return the full hidden-state
                sequence (B, L_enc, hidden_size) for downstream GHMM use.
        Returns:
            preds: (B, horizon)
            embeddings (optional): (B, L_enc, hidden_size) — causal hidden states
        """
        x = self.input_proj(batch["encoder_cont"])   # (B, L_enc, H)
        out, _ = self.lstm(x)                        # (B, L_enc, H)
        out = self.norm(out)
        out = self.drop(out)
        last = out[:, -1, :]                         # (B, H)
        preds = self.head(last)                      # (B, horizon)
        if return_embeddings:
            return preds, out                        # out: (B, L_enc, hidden_size)
        return preds

    # ------------------------------------------------------------------
    # Shared metric computation
    # ------------------------------------------------------------------
    def _compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        p = preds.reshape(-1)
        t = targets.reshape(-1)
        mask = torch.isfinite(p) & torch.isfinite(t)
        p, t = p[mask], t[mask]
        if len(p) == 0:
            return None
        mse  = torch.mean((p - t) ** 2)
        rmse = torch.sqrt(mse)
        mae  = torch.mean(torch.abs(p - t))
        ss_res = torch.sum((t - p) ** 2)
        ss_tot = torch.sum((t - t.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return mse, rmse, mae, r2

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = batch["target"].squeeze(-1)         # (B, H)

        # Skip corrupted batches
        if torch.isnan(targets).any() or torch.isnan(batch["encoder_cont"]).any():
            return None

        preds = self(batch)                           # (B, H)
        mask  = torch.isfinite(targets) & torch.isfinite(preds)
        loss  = torch.mean((preds[mask] - targets[mask]) ** 2)

        if torch.isnan(loss) or torch.isinf(loss):
            return None

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        metrics = self._compute_metrics(preds.detach(), targets.detach())
        if metrics:
            mse, rmse, *_ = metrics
            self.log("train_mse",  mse,  on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_rmse", rmse, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["target"].squeeze(-1)
        preds   = self(batch)
        mask    = torch.isfinite(targets) & torch.isfinite(preds)
        loss    = torch.mean((preds[mask] - targets[mask]) ** 2)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        metrics = self._compute_metrics(preds.detach(), targets.detach())
        if metrics:
            mse, rmse, mae, r2 = metrics
            self.log("val_mse",  mse,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_mae",  mae,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_r2",   r2,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        targets = batch["target"].squeeze(-1)
        preds   = self(batch)
        mask    = torch.isfinite(targets) & torch.isfinite(preds)
        loss    = torch.mean((preds[mask] - targets[mask]) ** 2)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        metrics = self._compute_metrics(preds.detach(), targets.detach())
        if metrics:
            mse, rmse, mae, r2 = metrics
            self.log("test_mse",  mse,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_mae",  mae,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test_r2",   r2,   on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
