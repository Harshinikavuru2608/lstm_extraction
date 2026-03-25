"""
Configuration file for MAMBA-CGM model (DDP multi-node version)
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: Path = Path("/users/PAS0536/harshinikavuru26/mydata/AI_READI/Data/3.0")
    processed_data_path: Path = Path("/users/PAS0536/harshinikavuru26/mydata/AI_READI/mamba_cgm/data/processed_data_3.0.parquet")

    # Resampling
    resample_freq: str = "5min"  # 5-minute cadence

    # Time series parameters
    context_length: int = 144  # 12 hours context (288 * 5min)
    horizon: int = 12  # 1 hour prediction (12 * 5min = 60min)

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Batch size (per GPU — effective batch = batch_size * num_nodes)
    batch_size: int = 256
    num_workers: int = 6

    # Feature columns
    cgm_col: str = "cgm_glucose"
    time_col: str = "timestamp"
    participant_col: str = "participant_id"

    # Static categorical features
    static_categoricals: List[str] = field(default_factory=lambda: ["participant_id", "clinical_site"])

    # Static real-valued features
    static_reals: List[str] = field(default_factory=lambda: ["age", "BMI", "wth"])

    # Time-varying known features (we know these in future)
    time_varying_known_reals: List[str] = field(default_factory=lambda: [
        "minute_of_day", "hour_of_day",
            "tod_sin", "tod_cos","heart_rate", "respiratory_rate", "stress_level", "movement",
    ])

    time_varying_known_categoricals: List[str] = field(default_factory=lambda: [
        "sleep_stage_state",
        "activity_name",
    ])

    # Time-varying unknown features (we don't know these in future)
    time_varying_unknown_reals: List[str] = field(default_factory=lambda: [
          "cgm_glucose", "cgm_lag_1", "cgm_lag_3", "cgm_lag_6", "cgm_diff_1", "cgm_diff_3",
            "cgm_rolling_mean_6", "cgm_rolling_std_6",
    ])
   

@dataclass
class MambaBlockConfig:
    """Configuration for MAMBA blocks (passed as mamba_block_config)"""
    d_state: int = 128
    d_conv: int = 4
    expand: int = 4
    headdim: int = 128
    ngroups: int = 1
    x_share_mode: str = "mean"
    se_reduce_ratio: int = 4
    return_hidden_attn: bool = False

    def to_dict(self):
        return {
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "ngroups": self.ngroups,
            "x_share_mode": self.x_share_mode,
            "se_reduce_ratio": self.se_reduce_ratio,
            "return_hidden_attn": self.return_hidden_attn,
        }


@dataclass
class Mamba2RuntimeConfig:
    """Runtime configuration for Mamba2MES encoder"""
    chunk_size: int = 128
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    learnable_init_states: bool = False
    D: Optional[float] = None

    def to_dict(self):
        return {
            "chunk_size": self.chunk_size,
            "dt_limit": self.dt_limit,
            "learnable_init_states": self.learnable_init_states,
            "D": self.D,
        }


@dataclass
class LightMambaConfig:
    """Configuration for lightweight MAMBA (simpler version)"""
    d_model: int = 64
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1


@dataclass
class VSNConfig:
    """Variable Selection Network configuration"""
    hidden_size: int = 128
    dropout: float = 0.1
    num_static: int = 1
    num_known_reals: int = 5
    num_known_cats: int = 1
    num_unknown_reals: int = 13


@dataclass
class GRNConfig:
    """Gated Residual Network configuration"""
    hidden_size: int = 128
    dropout: float = 0.1
    context_size: Optional[int] = None  # For static context


@dataclass
class ModelConfig:
    """Main model configuration"""
    # Architecture
    hidden_size: int = 128  # d_model
    num_mamba_layers: int = 4  # mamba_depth
    dropout: float = 0.1  # mamba_dropout
    attention_heads: int = 4

    # Output
    num_quantiles: int = 3  # [0.1, 0.5, 0.9]
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10

    # Components
    mamba: MambaBlockConfig = field(default_factory=MambaBlockConfig)
    mamba2_runtime: Mamba2RuntimeConfig = field(default_factory=Mamba2RuntimeConfig)
    light_mamba: LightMambaConfig = field(default_factory=LightMambaConfig)
    vsn: VSNConfig = field(default_factory=VSNConfig)
    grn: GRNConfig = field(default_factory=GRNConfig)

    # Checkpointing
    use_gradient_checkpointing: bool = True


@dataclass
class DDPConfig:
    """DDP-specific configuration"""
    num_nodes: int = 4
    gpus_per_node: int = 1
    strategy: str = "ddp"
    find_unused_parameters: bool = False
    # NCCL timeout in seconds (increase for large models/slow interconnects)
    nccl_timeout: int = 1800


@dataclass
class TrainingConfig:
    """Training configuration"""
    seed: int = 42
    max_epochs: int = 30
    accelerator: str = "gpu"
    devices: int = 1  # GPUs per node
    strategy: str = "ddp"
    precision: str = "16-mixed"  # bf16-mixed for A100

    # DDP
    ddp: DDPConfig = field(default_factory=DDPConfig)

    # Paths
    checkpoint_dir: Path = Path("../checkpoints_ddp_12_144")
    log_dir: Path = Path("./logs")

    # Logging
    log_every_n_steps: int = 500
    val_check_interval: float = 1.0


def get_default_config():
    """Get default configuration"""
    return {
        "data": DataConfig(),
        "model": ModelConfig(),
        "training": TrainingConfig(),
    }


def build_param_dict(data_cfg: DataConfig = None, model_cfg: ModelConfig = None,
                     training_cfg: TrainingConfig = None):
    """Build the param dict used by the training script from config dataclasses."""
    if data_cfg is None:
        data_cfg = DataConfig()
    if model_cfg is None:
        model_cfg = ModelConfig()
    if training_cfg is None:
        training_cfg = TrainingConfig()

    param = {
        "dataset": {
            "horizon": data_cfg.horizon,
            "context_length": data_cfg.context_length,
            "batch_size": data_cfg.batch_size,
        },
        "mamba_block": model_cfg.mamba.to_dict(),
        "mamba_tft_init": {
            "d_model": model_cfg.hidden_size,
            "mamba_depth": model_cfg.num_mamba_layers,
            "mamba_dropout": model_cfg.dropout,
            "learning_rate": model_cfg.learning_rate,
        },
        "mamba2_mes_runtime": model_cfg.mamba2_runtime.to_dict(),
    }

    return param
