import argparse
import logging
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
    TimeSeriesDataSet,
    LSTM
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.models.base_model import BaseModel

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
    parser = argparse.ArgumentParser(description="Train and compare TFT and LSTM forecast strategies")
    parser.add_argument("--csv_path", type=Path, default=CSV_PATH)
    parser.add_argument("--predicting", type=str, default='GHI', help="Target variable to predict (e.g., 'GHI', 't2m').")
    parser.add_argument("--model", type=str, default='tft', choices=['tft', 'lstm'], help="Model to train and evaluate.")
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
    predicting: str
) -> tuple[TimeSeriesDataSet, pd.DataFrame]:
    """Creates training and validation datasets with a proper time-based split."""
    validation_cutoff = df['time_idx'].max() - (365 * 24)
    training_df = df[df.time_idx <= validation_cutoff]
    validation_df = df[df.time_idx > validation_cutoff]

    logger.info(f"Training data from {training_df.datetime.min()} to {training_df.datetime.max()}")
    logger.info(f"Validation data from {validation_df.datetime.min()} to {validation_df.datetime.max()}")

    time_varying_known_categoricals = ['day_of_week', 'day_of_month', 'month', 'hour']
    time_varying_known_reals = ['time_idx', 'sin_doy', 'cos_doy']
    time_varying_unknown_reals = list(df.select_dtypes(include=np.number).columns)
    
    time_varying_unknown_reals.remove(predicting)
    if 'time_idx' in time_varying_unknown_reals:
        time_varying_unknown_reals.remove('time_idx')
    
    try:
        training_dataset = TimeSeriesDataSet(
            training_df,
            time_idx='time_idx',
            target=predicting,
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


def build_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
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

def build_lstm_model(training_dataset: TimeSeriesDataSet) -> LSTM:
    """Instantiates a standard LSTM model that uses the features from the dataset."""
    # CORRECT APPROACH: Instantiate the LSTM by explicitly passing the feature names
    # that are available in the TimeSeriesDataSet. This is the intended way.
    return LSTM(
        # Pass the feature names from the dataset to the model
        x_reals=training_dataset.reals,
        x_categoricals=training_dataset.categoricals,
        
        # Standard model hyperparameters
        hidden_size=64,
        n_layers=2,
        dropout=0.1,
        learning_rate=0.01,
        loss=SMAPE(),
        log_interval=10,
    )


def train_model(
    training_dataset: TimeSeriesDataSet,
    validation_df: pd.DataFrame,
    args: argparse.Namespace
) -> str:
    """Trains the selected model and returns the best checkpoint path."""
    val_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_df, predict=True, stop_randomization=True)
    train_loader = training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=args.batch_size * 10, num_workers=0)

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(monitor="val_loss", filename=f"{args.model}-{{epoch:02d}}-{{val_loss:.4f}}", save_top_k=1, mode="min")
    ]
    tb_logger = TensorBoardLogger("lightning_logs", name=f"{args.model}_{training_dataset.target}_run")
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=args.max_epochs,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=tb_logger,
    )
    
    if args.model == 'tft':
        model = build_tft_model(training_dataset)
    elif args.model == 'lstm':
        model = build_lstm_model(training_dataset)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader, val_loader)
    if lr_finder.suggestion():
        model.hparams.learning_rate = lr_finder.suggestion()
        logger.info(f"Using suggested learning rate: {model.hparams.learning_rate:.2e}")
    else:
        logger.warning("Could not find a learning rate. Using the default.")


    trainer.fit(model, train_loader, val_loader)
    return trainer.checkpoint_callback.best_model_path


def plot_strategy_comparison(results_df: pd.DataFrame, mae_day_ahead: float, mae_rolling: float, predicting: str, model_type: str):
    """
    Creates a high-quality, publication-ready figure comparing the two forecasting strategies.
    """
    logger.info(f"Generating publication-quality figure for strategy comparison using {model_type.upper()}...")
    plt.style.use("seaborn-v0_8-ticks")

    unit = "(W/m²)" if predicting == "GHI" else "(°C)"
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_hours = 168
    plot_df = results_df.iloc[:plot_hours]
    
    ax.plot(plot_df.index, plot_df['actual'], label=f'Actual {predicting}', color='black', linewidth=2.5, zorder=4)
    ax.plot(plot_df.index, plot_df['day_ahead_pred'], label=f'Day-Ahead (Static) Forecast (MAE: {mae_day_ahead:.2f})', color='#D55E00', linestyle='--', linewidth=2, zorder=3)
    ax.plot(plot_df.index, plot_df['rolling_pred'], label=f'Rolling Horizon (Adaptive) Forecast (MAE: {mae_rolling:.2f})', color='#0072B2', linewidth=2, zorder=5)
    
    if 'rolling_pred_p10' in plot_df.columns and 'rolling_pred_p90' in plot_df.columns:
        ax.fill_between(plot_df.index, plot_df['rolling_pred_p10'], plot_df['rolling_pred_p90'], color='#56B4E9', alpha=0.3, label='Rolling Horizon 10th-90th Percentile')

    ax.set_title(f'{model_type.upper()} Forecasting Strategy Comparison: Static vs. Adaptive ({predicting})', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{predicting} {unit}', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    fig.tight_layout()
    fig.savefig(f"{model_type}_strategy_comparison_{predicting}.png", dpi=300)
    plt.show()


def run_comparison_simulation(
    ckpt_path: str,
    full_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    args: argparse.Namespace
):
    """Simulates and compares Day-Ahead vs. Rolling Horizon forecasting for the given model."""
    logger.info(f"--- Starting Forecast Comparison Simulation for {args.predicting} with {args.model.upper()} ---")
    
    if args.model == 'tft':
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        predict_mode = "quantiles"
    elif args.model == 'lstm':
        model = LSTM.load_from_checkpoint(ckpt_path)
        predict_mode = "prediction"
    else:
        raise ValueError(f"Unknown model type: {args.model}")


    results = []
    start_time_idx = validation_df['time_idx'].min() + (args.days * 24)
    day_ahead_full_forecast = None

    for t in range(args.simulation_hours):
        current_time_idx = start_time_idx + t
        current_datetime = full_df.loc[full_df.time_idx == current_time_idx, 'datetime'].iloc[0]
        
        if t % 24 == 0:
            logger.info(f"Simulating hour {t+1}/{args.simulation_hours} ({current_datetime}) - Generating new Day-Ahead forecast...")
            prediction_input_df = full_df[
                (full_df.time_idx >= current_time_idx - args.max_encoder_length) &
                (full_df.time_idx < current_time_idx + args.max_prediction_length)
            ]
            day_ahead_full_forecast = model.predict(prediction_input_df, mode=predict_mode)

        if args.model == 'tft':
            day_ahead_pred = day_ahead_full_forecast[0, t % 24, 3].item()
        else:
            day_ahead_pred = day_ahead_full_forecast[0, t % 24].item()


        rolling_input_df = full_df[
            (full_df.time_idx >= current_time_idx - args.max_encoder_length) &
            (full_df.time_idx < current_time_idx + args.max_prediction_length)
        ]
        rolling_full_forecast = model.predict(rolling_input_df, mode=predict_mode)
        
        actual = full_df.loc[full_df.time_idx == current_time_idx, args.predicting].iloc[0]
        result_row = {'timestamp': current_datetime, 'actual': actual, 'day_ahead_pred': day_ahead_pred}

        if args.model == 'tft':
            result_row['rolling_pred'] = rolling_full_forecast[0, 0, 3].item()
            result_row['rolling_pred_p10'] = rolling_full_forecast[0, 0, 0].item()
            result_row['rolling_pred_p90'] = rolling_full_forecast[0, 0, 6].item()
        else:
            result_row['rolling_pred'] = rolling_full_forecast[0, 0].item()

        if args.predicting != "GHI":
            for key in ['actual', 'day_ahead_pred', 'rolling_pred', 'rolling_pred_p10', 'rolling_pred_p90']:
                if key in result_row:
                    result_row[key] -= 273.15
        
        results.append(result_row)

    results_df = pd.DataFrame(results).set_index('timestamp')
    mae_day_ahead = (results_df['actual'] - results_df['day_ahead_pred']).abs().mean()
    mae_rolling = (results_df['actual'] - results_df['rolling_pred']).abs().mean()

    logger.info(f"--- Simulation Results for {args.predicting} ({args.model.upper()}) ---")
    logger.info(f"Day-Ahead (Static) Forecast MAE: {mae_day_ahead:.2f}")
    logger.info(f"Rolling Horizon (Adaptive) Forecast MAE: {mae_rolling:.2f}")
    
    if mae_day_ahead > 0:
        improvement = ((mae_day_ahead - mae_rolling) / mae_day_ahead) * 100
        logger.info(f"Rolling Horizon Accuracy Improvement: {improvement:.2f}%")

    plot_strategy_comparison(results_df, mae_day_ahead, mae_rolling, args.predicting, args.model)


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    df = load_data(args.csv_path)
    df = feature_processing(df)

    training_dataset, validation_df = create_datasets(
        df, args.max_encoder_length, args.max_prediction_length, args.predicting
    )

    if args.train:
        logger.info(f"--- Starting Model Training for {args.model.upper()} on {args.predicting} ---")
        best_ckpt_path = train_model(training_dataset, validation_df, args)
        logger.info(f"Training complete. Best model saved to: {best_ckpt_path}")
    else:
        ckpt_dir = Path("lightning_logs") / f"{args.model}_{args.predicting}_run"
        try:
            checkpoints = list(ckpt_dir.rglob(f"{args.model}-*.ckpt"))
            if not checkpoints:
                 raise FileNotFoundError("No checkpoints found.")
            best_ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Skipping training. Using latest model: {best_ckpt_path}")
        except (FileNotFoundError, IndexError):
             logger.error(f"No pre-trained model found for {args.model.upper()} on {args.predicting} in {ckpt_dir}. Please train a model first using the --train flag.")
             sys.exit(1)


    try:
        run_comparison_simulation(
            ckpt_path=str(best_ckpt_path),
            full_df=df,
            validation_df=validation_df,
            args=args
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

