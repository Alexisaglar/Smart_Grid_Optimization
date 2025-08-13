import argparse
import logging
from os import wait
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss

# --- Configuration ---
CSV_PATH = Path("data/processed_data/combined_data.csv")
MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 24
BATCH_SIZE = 128
MAX_EPOCHS = 150 
TRAIN_MODEL = False # Set to True to retrain the model

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and compare TFT forecast strategies")
    parser.add_argument("--csv_path", type=Path, default=CSV_PATH)
    parser.add_argument("--predicting", type=str, default='GHI')
    parser.add_argument("--days", type=int, default=168)
    parser.add_argument("--simulation_hours", type=int, default=168)
    parser.add_argument("--max_encoder_length", type=int, default=MAX_ENCODER_LENGTH)
    parser.add_argument("--max_prediction_length", type=int, default=MAX_PREDICTION_LENGTH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--train", action="store_true", default=TRAIN_MODEL,
        help="If set, run model training; otherwise, load a pre-trained model."
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    """Loads and performs initial processing of the data."""
    try:
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df["Unnamed: 0"])
        df = df.drop(columns=['Unnamed: 0'])
        df = df.sort_values("datetime").reset_index(drop=True)
        df['time_idx'] = df.index
    except Exception as e:
        logger.exception(f"Error loading or processing data from {csv_path}")
        raise
    return df


def feature_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Creates calendar, wind, and lag features."""
    try:
        df['month'] = df['datetime'].dt.month.astype(str).astype("category")
        df["hour"] = df['datetime'].dt.hour.astype(str).astype("category")
        df['day_of_month'] = df['datetime'].dt.day.astype(str).astype("category")
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype(str).astype("category")

        doy = df['datetime'].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['wind_dir'] = np.degrees(np.arctan2(df['v10'], df['u10'])) % 360
        df['sin_wdir'] = np.sin(np.deg2rad(df['wind_dir']))
        df['cos_wdir'] = np.cos(np.deg2rad(df['wind_dir']))

        df['t2m'] = df['t2m'] + 273.15
        df['group_id'] = 'site0-NCL'
        df['group_id'] = df['group_id'].astype('category')

        df = df.sort_values(['group_id', 'time_idx'])
        df['t2m_lag24'] = df.groupby('group_id', observed=True)['t2m'].shift(24)
        df = df.dropna(subset=['t2m_lag24']).reset_index(drop=True)
        df['time_idx'] = df.index

    except Exception:
        logger.exception("Error while processing features")
        raise
    return df


def create_datasets(
    df: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
) -> tuple[TimeSeriesDataSet, pd.DataFrame]:
    """Creates training and validation datasets with a proper time-based split."""
    validation_cutoff = df['time_idx'].max() - (365 * 24)
    training_df = df[df.time_idx <= validation_cutoff]
    validation_df = df[df.time_idx > validation_cutoff]

    logger.info(f"Training data from {training_df.datetime.min()} to {training_df.datetime.max()}")
    logger.info(f"Validation data from {validation_df.datetime.min()} to {validation_df.datetime.max()}")

    time_varying_known_categoricals = ['day_of_week', 'day_of_month', 'month', 'hour']
    time_varying_known_reals = [
        'time_idx','sin_doy','cos_doy'
    ]
    time_varying_unknown_reals = [
        # "t2m", # The actual temperature is now an unknown input
        # "t2m_lag24",
        "u10", "v10", "wind_speed", "sin_wdir", "cos_wdir",
        "tp",
        "TOA", # Top-of-Atmosphere is theoretically known, but often grouped with weather
        "Clear sky BNI", "Clear sky GHI", "Clear sky BHI", "Clear sky DHI", "Clear sky BNI",
        "BHI", "DHI", "BNI", # The other real irradiance components
        "GHI", 
    ]

    try:
        training_dataset = TimeSeriesDataSet(
            training_df,
            time_idx='time_idx',
            # target='GHI',
            target='t2m',
            group_ids=['group_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    except Exception:
        logger.exception("Error creating TimeSeriesDataSet")
        raise
        
    return training_dataset, validation_df


def build_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Instantiates a powerful TFT model."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=64,
        loss=QuantileLoss(),
        log_interval=10,
        optimizer='ranger',
    )


def train_model(
    training_dataset: TimeSeriesDataSet,
    validation_df: pd.DataFrame,
    args: argparse.Namespace
) -> str:
    """Trains the model and returns the best checkpoint path."""
    val_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_df)
    train_loader = training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=args.batch_size * 10, num_workers=0)

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(monitor="val_loss", filename="tft-{epoch:02d}-{val_loss:.4f}", save_top_k=1, mode="min")
    ]
    tb_logger = TensorBoardLogger("lightning_logs", name="tft_irradiance_run")
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=args.max_epochs,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=tb_logger,
    )
    
    model = build_model(training_dataset)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader, val_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    logger.info(f"Using suggested learning rate: {model.hparams.learning_rate:.2e}")

    trainer.fit(model, train_loader, val_loader)
    return trainer.checkpoint_callback.best_model_path


def run_comparison_simulation(
    ckpt_path: str,
    full_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int,
    predicting: str,
    days: int,
    simulation_hours: int,
):
    """Simulates and compares Day-Ahead vs. Rolling Horizon forecasting."""
    logger.info("--- Starting Forecast Comparison Simulation ---")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    results = []
    # start_time_idx = validation_df['time_idx'].min()
    # start_time_idx = validation_df['time_idx'].min()
    days = days
    start_time_idx = 87624 + days * 24

    for t in range(simulation_hours):
        current_time_idx = start_time_idx + t
        current_datetime = validation_df.loc[validation_df.time_idx == current_time_idx, 'datetime'].iloc[0]
        
        # --- 1. Day-Ahead (Static) Forecast ---
        if t % 24 == 0:
            logger.info(f"Simulating hour {t+1}/{simulation_hours} ({current_datetime}) - Generating new Day-Ahead forecast...")
            # CORRECTED SLICE: Must include history + future knowns
            prediction_input_df = full_df[
                (full_df.time_idx >= current_time_idx - max_encoder_length) &
                (full_df.time_idx < current_time_idx + max_prediction_length)
            ]
            day_ahead_full_forecast = model.predict(prediction_input_df)

        day_ahead_pred = day_ahead_full_forecast[0, t % 24].item()

        # --- 2. Rolling Horizon (Adaptive) Forecast ---
        # CORRECTED SLICE: Must include history + future knowns
        rolling_input_df = full_df[
            (full_df.time_idx >= current_time_idx - max_encoder_length) &
            (full_df.time_idx < current_time_idx + max_prediction_length)
        ]
        rolling_pred = model.predict(rolling_input_df)[0, 0].item()

        # --- 3. Get Actual Value ---
        # actual = validation_df.loc[validation_df.time_idx == current_time_idx, 'GHI'].iloc[0]
        actual = validation_df.loc[validation_df.time_idx == current_time_idx, predicting].iloc[0]

        if predicting == "GHI":
            results.append({
                'timestamp': current_datetime,
                'actual': actual,
                'day_ahead_pred': day_ahead_pred,
                'rolling_pred': rolling_pred
            })

        else:
            results.append({
                'timestamp': current_datetime,
                'actual': actual - 273.15,
                'day_ahead_pred': day_ahead_pred - 273.15,
                'rolling_pred': rolling_pred - 273.15
            })

    # --- 4. Analyze and Plot Results ---
    results_df = pd.DataFrame(results).set_index('timestamp')
    mae_day_ahead = (results_df['actual'] - results_df['day_ahead_pred']).abs().mean()
    mae_rolling = (results_df['actual'] - results_df['rolling_pred']).abs().mean()

    logger.info("--- Simulation Results ---")
    logger.info(f"Day-Ahead Forecast MAE: {mae_day_ahead:.2f}")
    logger.info(f"Rolling Horizon Forecast MAE: {mae_rolling:.2f}")
    
    if mae_day_ahead > 0:
        improvement = ((mae_day_ahead - mae_rolling) / mae_day_ahead) * 100
        logger.info(f"Rolling Horizon Improvement: {improvement:.2f}%")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))
    results_df['actual'].plot(ax=ax, label='Actual GHI', color='black', linewidth=2)
    results_df['day_ahead_pred'].plot(ax=ax, label=f'Day-Ahead (MAE: {mae_day_ahead:.2f})', linestyle='--', color='red')
    results_df['rolling_pred'].plot(ax=ax, label=f'Rolling Horizon (MAE: {mae_rolling:.2f})', linestyle=':', color='blue', linewidth=2.5)
    
    ax.set_title('Forecast Accuracy: Day-Ahead vs. Rolling Horizon', fontsize=16)
    ax.set_ylabel('Global Horizontal Irradiance (GHI)')
    ax.set_xlabel('Date')
    plt.ylim([0,(np.max(results_df['actual']) * 2)])
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("forecast_comparison.png")
    plt.show()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    df = load_data(args.csv_path)
    df = feature_processing(df)

    training_dataset, validation_df = create_datasets(
        df, args.max_encoder_length, args.max_prediction_length
    )

    if args.train:
        logger.info("--- Starting Model Training ---")
        best_ckpt_path = train_model(training_dataset, validation_df, args)
        logger.info(f"Training complete. Best model saved to: {best_ckpt_path}")
    else:


        # Make sure this path points to your best trained model
        # You can find this path in the output after running with --train
        if args.predicting == "GHI":
            best_ckpt_path = "models/new_tft_irradiance.ckpt"
        else:
            best_ckpt_path = "models/new_tft_temperature.ckpt"
        logger.info(f"Skipping training. Using existing model: {best_ckpt_path}")

    try:
        run_comparison_simulation(
            ckpt_path=best_ckpt_path,
            full_df=df,
            validation_df=validation_df,
            max_encoder_length=args.max_encoder_length,
            max_prediction_length=args.max_prediction_length,
            days=args.days,
            predicting=args.predicting,
            simulation_hours=args.simulation_hours,
        )
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at '{best_ckpt_path}'. Please train a model first or correct the path.")
    except Exception:
        logger.exception("An error occurred during the comparison simulation.")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception("Script terminated with errors")
        sys.exit(1)

