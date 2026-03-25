"""
Data preprocessing module for CGM + multimodal data
Handles loading, resampling to 5-minute cadence, and feature engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    Preprocessor for CGM and wearable sensor data.
    Resamples all data to 5-minute intervals and merges into unified dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        resample_freq: str = "5min",
        min_participant_hours: int = 24,  # Minimum hours of data per participant
    ):
        self.data_dir = Path(data_dir)
        self.resample_freq = resample_freq
        self.min_participant_hours = min_participant_hours
        self.freq_minutes = int(resample_freq.replace("min", ""))

    def load_cgm(self) -> pd.DataFrame:
        """Load and preprocess CGM data"""
        print("Loading CGM data...")
        df = pd.read_csv(self.data_dir / "cgm_3.0.csv")

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["start_time"], utc=True)
        df = df.rename(columns={"blood_glucose": "cgm_glucose"})
        df = df[["participant_id", "timestamp", "cgm_glucose"]]

        # Convert to numeric (handles string values)
        df["cgm_glucose"] = pd.to_numeric(df["cgm_glucose"], errors="coerce")

        # Clip invalid values instead of removing
        df["cgm_glucose"] = df["cgm_glucose"].clip(lower=20, upper=600)

        # Drop NaN values
        df = df.dropna(subset=["cgm_glucose"])

        print(f"  CGM: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        print(f"  Total participants with CGM values after cleaning: {df['participant_id'].nunique()}")
        return df

    def load_heart_rate(self) -> pd.DataFrame:
        """Load and preprocess heart rate data"""
        print("Loading heart rate data...")
        df = pd.read_csv(self.data_dir / "heart_rate_3.0.csv")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns={"heart_rate_bpm": "heart_rate"})
        df = df[["participant_id", "timestamp", "heart_rate"]]

        # Convert to numeric
        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")

        # Clip invalid values (0.0 indicates missing, set minimum to 1)
        df["heart_rate"] = df["heart_rate"].clip(lower=1)

        # Drop NaN values
        df = df.dropna(subset=["heart_rate"])

        print(f"  Heart Rate: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        return df

    def load_respiratory_rate(self) -> pd.DataFrame:
        """Load and preprocess respiratory rate data"""
        print("Loading respiratory rate data...")
        df = pd.read_csv(self.data_dir / "respiratory_rate_3.0.csv")

        # Handle different timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns={"respiratory_rate_bpm": "respiratory_rate"})
        df = df[["participant_id", "timestamp", "respiratory_rate"]]

        # Convert to numeric
        df["respiratory_rate"] = pd.to_numeric(df["respiratory_rate"], errors="coerce")

        # Clip invalid values
        df["respiratory_rate"] = df["respiratory_rate"].clip(lower=5, upper=60)

        # Drop NaN values
        df = df.dropna(subset=["respiratory_rate"])

        print(f"  Respiratory Rate: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        return df

    def load_stress(self) -> pd.DataFrame:
        """Load and preprocess stress data"""
        print("Loading stress data...")
        df = pd.read_csv(self.data_dir / "stress_3.0.csv")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns={"stress": "stress_level"})
        df = df[["participant_id", "timestamp", "stress_level"]]

        # Convert to numeric
        df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")

        # Clip invalid values (negative values indicate missing, set minimum to 0)
        df["stress_level"] = df["stress_level"].clip(lower=0)

        # Drop NaN values
        df = df.dropna(subset=["stress_level"])

        print(f"  Stress: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        return df

    def load_activity(self) -> pd.DataFrame:
        """Load and preprocess physical activity data (event-based)"""
        print("Loading physical activity data...")
        df = pd.read_csv(self.data_dir / "physical_activity_3.0.csv")

        df["start_timestamp"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_timestamp"] = pd.to_datetime(df["end_time"], utc=True)

        # Keep activity_name as categorical string (label-encoded later in dataset)
        df["activity_name"] = df["activity_name"].fillna("sedentary")

        # Convert movement to numeric
        df["movement"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)

        print(f"  Activity: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        return df[["participant_id", "start_timestamp", "end_timestamp", "activity_name", "movement"]]

    def load_sleep(self) -> pd.DataFrame:
        """Load and preprocess sleep data (event-based)"""
        print("Loading sleep data...")
        df = pd.read_csv(self.data_dir / "sleep_3.0.csv")

        df["start_timestamp"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_timestamp"] = pd.to_datetime(df["end_time"], utc=True)

        # Keep sleep_stage_state as categorical string (label-encoded later in dataset)
        df["sleep_stage_state"] = df["sleep_stage_state"].fillna("awake")

        print(f"  Sleep: {len(df):,} rows, {df['participant_id'].nunique()} participants")
        return df[["participant_id", "start_timestamp", "end_timestamp", "sleep_stage_state"]]

    def load_participant_metadata(self) -> pd.DataFrame:
        """Load and preprocess participant metadata with static features"""
        print("Loading participant metadata...")

        # Helper function to safely parse measurements column
        def parse_measurements(x):
            if isinstance(x, dict):
                return x
            if pd.isna(x):
                return {}
            s = str(x)
            # evaluate with a safe globals dict, binding `nan` -> np.nan
            try:
                result = eval(s, {"__builtins__": {}}, {"nan": np.nan})
            except Exception:
                # fallback: turn `nan:` (as a key) into a quoted key so literal_eval works
                import ast
                import re
                s2 = re.sub(r'(?<![A-Za-z0-9_])nan\s*:', "'nan':", s)
                try:
                    result = ast.literal_eval(s2)
                except Exception:
                    return {}
            
            # If result is a list of dicts, merge them into one dict
            if isinstance(result, list):
                merged = {}
                for item in result:
                    if isinstance(item, dict):
                        merged.update(item)
                return merged
            
            return result if isinstance(result, dict) else {}

        # Load the CSV file
        df = pd.read_csv(self.data_dir / "final_dataset.csv")
        df = df.rename(columns={'measurements_list':'measurements'})

        # Parse measurements column once
        if 'measurements' in df.columns:
            mseries = df["measurements"].apply(parse_measurements)
        else:
            mseries = pd.Series([{} for _ in range(len(df))])

        # Create metadata DataFrame
        metadata = pd.DataFrame()
        metadata["participant_id"] = df["person_id"]

        # Add categorical columns
        # Note: clinical_site not in final_df.csv, use unknown as default
        metadata["clinical_site"] = "unknown"
        metadata["study_group"] = df["study_group"] if "study_group" in df.columns else "unknown"

        # Add age
        metadata["age"] = pd.to_numeric(df["age"], errors="coerce") if "age" in df.columns else np.nan

        # Extract measurements from the parsed dictionary
        metadata["BMI"] = mseries.apply(lambda d: d.get("Body mass index", np.nan))
        metadata["wth"] = mseries.apply(lambda d: d.get("Waist to height ratio", np.nan))
        metadata["weight"] = mseries.apply(lambda d: d.get("Body weight", np.nan))

        # Optional: extract height for potential BMI calculation
        metadata["height"] = mseries.apply(lambda d: d.get("Body height", np.nan))
        metadata["waist"] = mseries.apply(lambda d: d.get("Waist Circumference", np.nan))

        # Calculate BMI if not present but height and weight are available
        bmi_missing = metadata["BMI"].isna()
        has_height_weight = metadata["height"].notna() & metadata["weight"].notna()
        if bmi_missing.any() and has_height_weight.any():
            # BMI = weight(kg) / (height(m))^2
            # Convert height from cm to m if needed
            metadata.loc[bmi_missing & has_height_weight, "BMI"] = (
                metadata.loc[bmi_missing & has_height_weight, "weight"] /
                (metadata.loc[bmi_missing & has_height_weight, "height"] / 100) ** 2
            )

        # Calculate waist-to-height ratio if not present but both measurements are available
        wth_missing = metadata["wth"].isna()
        has_waist_height = metadata["waist"].notna() & metadata["height"].notna()
        if wth_missing.any() and has_waist_height.any():
            metadata.loc[wth_missing & has_waist_height, "wth"] = (
                metadata.loc[wth_missing & has_waist_height, "waist"] /
                metadata.loc[wth_missing & has_waist_height, "height"]
            )

        # Remove duplicates - keep first occurrence per participant
        metadata = metadata.drop_duplicates(subset=["participant_id"], keep="first")

        print(f"  Metadata: {len(metadata):,} participants")
        print(f"  Columns: {list(metadata.columns)}")
        print(f"  BMI available: {metadata['BMI'].notna().sum()} participants")
        print(f"  Waist-to-height available: {metadata['wth'].notna().sum()} participants")

        return metadata

    def _resample_point_data(
        self,
        df: pd.DataFrame,
        value_col: str,
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """Resample point-based time series data to 5-minute intervals"""
        resampled_dfs = []

        for pid in df["participant_id"].unique():
            pdf = df[df["participant_id"] == pid].copy()
            pdf = pdf.set_index("timestamp").sort_index()

            # Resample to 5-minute intervals
            resampled = pdf[value_col].resample(self.resample_freq).agg(agg_func)
            resampled = resampled.to_frame().reset_index()
            resampled["participant_id"] = pid
            resampled_dfs.append(resampled)

        return pd.concat(resampled_dfs, ignore_index=True)

    def _expand_event_data(
        self,
        df: pd.DataFrame,
        value_col: str,
        time_grid: pd.DataFrame,
    ) -> pd.DataFrame:
        """Expand event-based data (with start/end times) to time grid"""
        expanded_dfs = []

        for pid in time_grid["participant_id"].unique():
            # Get time grid for this participant
            pgrid = time_grid[time_grid["participant_id"] == pid][["timestamp"]].copy()

            # Get events for this participant
            events = df[df["participant_id"] == pid].copy()

            if len(events) == 0:
                pgrid[value_col] = np.nan
                pgrid["participant_id"] = pid
                expanded_dfs.append(pgrid)
                continue

            # For each timestamp, find the event it falls within
            values = []
            for ts in pgrid["timestamp"]:
                mask = (events["start_timestamp"] <= ts) & (events["end_timestamp"] > ts)
                matching = events[mask]
                if len(matching) > 0:
                    values.append(matching[value_col].iloc[0])
                else:
                    values.append(np.nan)

            pgrid[value_col] = values
            pgrid["participant_id"] = pid
            expanded_dfs.append(pgrid)

        return pd.concat(expanded_dfs, ignore_index=True)

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        # Extract time components
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Cyclical encoding
        df["tod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
        df["tod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def _create_cgm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create CGM-derived features (lagged, differenced, rolling)"""
        df = df.copy()

        # Sort by participant and time
        df = df.sort_values(["participant_id", "timestamp"])

        # Group by participant for feature engineering
        features = []
        for pid in tqdm(df["participant_id"].unique(), desc="Creating CGM features"):
            pdf = df[df["participant_id"] == pid].copy()

            # Lagged features
            pdf["cgm_lag_1"] = pdf["cgm_glucose"].shift(1)   # 5 min ago
            pdf["cgm_lag_3"] = pdf["cgm_glucose"].shift(3)   # 15 min ago
            pdf["cgm_lag_6"] = pdf["cgm_glucose"].shift(6)   # 30 min ago
            pdf["cgm_lag_12"] = pdf["cgm_glucose"].shift(12) # 1 hour ago

            # Difference features
            pdf["cgm_diff_1"] = pdf["cgm_glucose"].diff(1)
            pdf["cgm_diff_3"] = pdf["cgm_glucose"].diff(3)
            pdf["cgm_diff_6"] = pdf["cgm_glucose"].diff(6)

            # Rolling features
            pdf["cgm_rolling_mean_6"] = pdf["cgm_glucose"].rolling(6, min_periods=1).mean()
            pdf["cgm_rolling_std_6"] = pdf["cgm_glucose"].rolling(6, min_periods=1).std()
            pdf["cgm_rolling_mean_12"] = pdf["cgm_glucose"].rolling(12, min_periods=1).mean()

            # Rate of change
            pdf["cgm_roc_6"] = (pdf["cgm_glucose"] - pdf["cgm_lag_6"]) / 6  # mg/dL per 5min

            features.append(pdf)

        return pd.concat(features, ignore_index=True)

    def _create_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create integer time index for each participant"""
        df = df.copy()
        df = df.sort_values(["participant_id", "timestamp"])

        # Create sequential time index per participant
        df["time_idx"] = df.groupby("participant_id").cumcount()

        return df

    def process(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Main processing pipeline:
        1. Load all data sources
        2. Resample to 5-minute intervals
        3. Merge all sources
        4. Create features
        5. Clean and validate
        """
        print("=" * 60)
        print("Starting data preprocessing pipeline")
        print("=" * 60)

        # 1. Load all data
        cgm_df = self.load_cgm()
        hr_df = self.load_heart_rate()
        rr_df = self.load_respiratory_rate()
        stress_df = self.load_stress()
        activity_df = self.load_activity()
        sleep_df = self.load_sleep()
        metadata_df = self.load_participant_metadata()

        # 2. Find common participants
        common_pids = set(cgm_df["participant_id"].unique())
        common_pids &= set(hr_df["participant_id"].unique())
        print(f"\nParticipants with CGM + HR: {len(common_pids)}")

        # Filter to common participants
        cgm_df = cgm_df[cgm_df["participant_id"].isin(common_pids)]

        # 3. Resample CGM to 5-minute intervals (this is our base grid)
        print("\nResampling data to 5-minute intervals...")
        cgm_resampled = self._resample_point_data(cgm_df, "cgm_glucose", "mean")

        # 4. Create time grid from CGM data
        time_grid = cgm_resampled[["participant_id", "timestamp"]].copy()

        # 5. Resample point data sources
        hr_resampled = self._resample_point_data(
            hr_df[hr_df["participant_id"].isin(common_pids)],
            "heart_rate", "mean"
        )
        rr_resampled = self._resample_point_data(
            rr_df[rr_df["participant_id"].isin(common_pids)],
            "respiratory_rate", "mean"
        )
        stress_resampled = self._resample_point_data(
            stress_df[stress_df["participant_id"].isin(common_pids)],
            "stress_level", "mean"
        )

        # 6. Expand event-based data to time grid
        print("Expanding event-based data...")
        activity_expanded = self._expand_event_data(
            activity_df[activity_df["participant_id"].isin(common_pids)],
            "activity_name", time_grid
        )
        steps_expanded = self._expand_event_data(
            activity_df[activity_df["participant_id"].isin(common_pids)],
            "movement", time_grid
        )
        sleep_expanded = self._expand_event_data(
            sleep_df[sleep_df["participant_id"].isin(common_pids)],
            "sleep_stage_state", time_grid
        )

        # 7. Merge all data sources
        print("Merging data sources...")
        merged = cgm_resampled.copy()

        # Merge point data
        for df, col in [
            (hr_resampled, "heart_rate"),
            (rr_resampled, "respiratory_rate"),
            (stress_resampled, "stress_level"),
        ]:
            merged = merged.merge(
                df[["participant_id", "timestamp", col]],
                on=["participant_id", "timestamp"],
                how="left"
            )

        # Merge event data
        for df, col in [
            (activity_expanded, "activity_name"),
            (steps_expanded, "movement"),
            (sleep_expanded, "sleep_stage_state"),
        ]:
            merged = merged.merge(
                df[["participant_id", "timestamp", col]],
                on=["participant_id", "timestamp"],
                how="left"
            )

        # 8. Merge participant metadata (static features)
        print("Merging participant metadata...")
        merged = merged.merge(
            metadata_df,
            on="participant_id",
            how="left"
        )

        # 9. Create time-based features
        print("Creating time features...")
        merged = self._create_time_features(merged)

        # 10. Create CGM-derived features
        print("Creating CGM-derived features...")
        merged = self._create_cgm_features(merged)

        # 11. Create time index
        merged = self._create_time_index(merged)

        # 12. Handle missing values
        print("Handling missing values...")

        # Fill missing categorical event data with defaults
        merged["sleep_stage_state"] = merged["sleep_stage_state"].fillna("awake")
        merged["activity_name"] = merged["activity_name"].fillna("sedentary")

        # Forward fill categorical event data (within reasonable window)
        for cat_col in ["sleep_stage_state", "activity_name"]:
            merged[cat_col] = merged.groupby("participant_id")[cat_col].ffill(limit=6)

        # Forward fill for sensor data (within reasonable window)
        sensor_cols = ["heart_rate", "respiratory_rate", "stress_level"]
        merged[sensor_cols] = merged.groupby("participant_id")[sensor_cols].ffill(limit=6)

        # Fill remaining NaN in lagged features
        lag_cols = [c for c in merged.columns if "lag" in c or "diff" in c or "rolling" in c or "roc" in c]
        merged[lag_cols] = merged.groupby("participant_id")[lag_cols].bfill(limit=12)

        # Handle missing static categorical features with defaults
        if "clinical_site" in merged.columns:
            merged["clinical_site"] = merged["clinical_site"].fillna("unknown")
        if "study_group" in merged.columns:
            merged["study_group"] = merged["study_group"].fillna("unknown")

        # For static real features, forward fill within participant (they should be constant)
        static_reals = ["age", "BMI", "wth"]
        for col in static_reals:
            if col in merged.columns:
                merged[col] = merged.groupby("participant_id")[col].ffill().bfill()

        # 13. Filter participants with minimum data
        print("Filtering participants by data availability...")
        min_rows = self.min_participant_hours * (60 // self.freq_minutes)
        valid_pids = merged.groupby("participant_id").size()
        valid_pids = valid_pids[valid_pids >= min_rows].index
        merged = merged[merged["participant_id"].isin(valid_pids)]

        # 14. Drop rows with missing CGM (our target)
        merged = merged.dropna(subset=["cgm_glucose"])

        # 15. Final cleanup
        merged = merged.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)

        print("=" * 60)
        print(f"Final dataset: {len(merged):,} rows, {merged['participant_id'].nunique()} participants")
        print(f"Columns: {list(merged.columns)}")
        print("=" * 60)

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(save_path, index=False)
            print(f"Saved to {save_path}")

        return merged


def preprocess_data(
    data_dir: str = "/users/PAS0536/harshinikavuru26/mydata/AI_READI/Data/3.0",
    save_path: str = None,
) -> pd.DataFrame:
    """Convenience function to preprocess data"""
    preprocessor = DataPreprocessor(Path(data_dir))
    return preprocessor.process(save_path=save_path)


if __name__ == "__main__":
    # Run preprocessing
    df = preprocess_data(
        save_path="/users/PAS0536/harshinikavuru26/mydata/AI_READI/mamba_cgm/data/processed_data_3.0.parquet"
    )
    print(df.head())
    print(df.info())
