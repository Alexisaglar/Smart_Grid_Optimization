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
    Baseline
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from torch.utils.data.dataloader import DataLoader

# Configuration constants 
CSV_PATH = Path("data/processed_data/combined_data.csv")
MAX_ENCODER_LENGTH = 24
MAX_PREDICTION_LENGTH = 24
BATCH_SIZE = 128
MAX_EPOCHS = 150
DEBUG = True
TRAIN = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser(
        description="Train and interact with TFT forecasts"
    )
    parser.add_argument(
        "--csv_path", type=Path, default=CSV_PATH,
        help='Path to your preprocessed weather CSV'
    )
    parser.add_argument(
        "--max_encoder_length", type=int, default=MAX_ENCODER_LENGTH
    )
    parser.add_argument(
        "--max_prediction_length", type=int, default=MAX_PREDICTION_LENGTH,
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
    )
    parser.add_argument(
        "--max_epochs", type=int, default=MAX_EPOCHS,
    )
    parser.add_argument(
        "--train", action="store_true",
        help="If set, run model training; otherwise skip to DEBUG/plot only"
    )
    parser.add_argument(
        "--plot_date", type=str, default=None,
        help="Datetime (YYYY-MM-DD HH:MM) to produce history+forecast for"
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load clean CSV to convert it to DataFrame
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e: 
        logger.error(f"CSV file not found at {csv_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file")
        raise

    try:
        df['datetime']= pd.to_datetime(df["Unnamed: 0"])
        df = df.drop(columns=['Unnamed: 0'])
    except Exception as e:
        logger.error(f"Error processing datetime column")

    df['time_idx'] = np.arange(len(df))
    return df

def feature_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar, wind and lag features based on datetime
    """
    try:
        # Categorical time features
        df['month'] = df['datetime'].dt.year * 12 + df['datetime'].dt.month
        df['month'] -= df['month'].min()
        df['month'] = df['month'].astype(str).astype("category")

        df["hour"] = df['datetime'].dt.hour.astype(str).astype("category")

        df['day'] = df['datetime'].dt.year * 365 + df['datetime'].dt.day
        df['day'] -= df['day'].min()
        df['day'] = df['day'].astype(str).astype("category")

        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_week'] = df['day_of_week'].astype(str).astype("category")

        # seasonal cycle
        doy = df['datetime'].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        # wind features
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['wind_dir'] = np.degrees(np.arctan2(df['v10'], df['u10'])) % 360
        df['sin_wdir'] = np.sin(np.deg2rad(df['wind_dir']))
        df['cos_wdir'] = np.cos(np.deg2rad(df['wind_dir']))

        # Convert to Kelvin from Celsius as TFT doesn't accept negative values
        df['t2m'] = df['t2m'] + 273.15 

        # single site group id
        df['group_id'] = 'site0-NCL'
        df['group_id'] = df['group_id'].astype('category')

        # lag feature
        df = df.sort_values(['group_id', 'time_idx'])
        df['t2m_lag24'] = df.groupby('group_id')['t2m'].shift(24)
        df = df.dropna(subset=['t2m_lag24']).reset_index(drop=True)
    except Exception:
        logger.exception("Error while processing new categorical features")
        raise

    print(f'DF processed:\n{df.describe()}')
    return df

def create_datasets(
    df: pd.DataFrame, 
    max_encoder_length: int, 
    max_prediction_length: int,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Create Pytorch forecasting train and validation datasets
    """
    static_categoricals=[]
    static_reals = []
    time_varying_known_categoricals = ['day_of_week', 'day', 'month', 'hour']
    time_varying_known_reals=[
        'time_idx', 'TOA', 'Clear sky GHI', 'Clear sky BHI',
            'Clear sky DHI', 'Clear sky BNI', 'sin_doy', 'cos_doy'
    ]
    time_varying_unknown_reals = [
        "t2m_lag24", "u10", "v10", "wind_speed",
        "sin_wdir", "cos_wdir", "tp", "GHI", "BHI", "DHI", "BNI",
    ]
    cutoff = df['time_idx'].max() - max_prediction_length
    training_df = df[df.time_idx <= cutoff]

    print(df.head())
    try:
        training_dataset = TimeSeriesDataSet(
            training_df,
            time_idx='time_idx',
            target='t2m',
            group_ids=['group_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        print(training_dataset)
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, training_df, predict=True, stop_randomization=True,
        )

    except Exception:
        logger.exception("Error creating TimeSeriesDataSet")
    
    return training_dataset, validation_dataset

def get_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    batch_size: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convert datasets into PyTorch dataloaders
    """
    try:
        train_loader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation_dataset.to_dataloader(
                train=False, batch_size=batch_size * 10, num_workers=0
        )

    except Exception:
        logger.exception("Error while creating dataloaders.")
        raise
    return train_loader, val_loader

    

def build_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """
    Instantiate TFT model from dataset
    """
    try:
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            log_interval=10,
            optimizer='ranger',
            # reduce_on_plateau_patience=4,
        )
        logger.info(f"Model size: {tft.size()/1e3:.1f}k parameters")

    except Exception:
        logger.exception("Error building TemporalFusionTransformer")

    return tft

def find_learning_rate(
    trainer: pl.Trainer,
    model: TemporalFusionTransformer,
    train_loader: TimeSeriesDataSet,
    val_loader: TimeSeriesDataSet,
) -> float:
    """
    Finding optimal LR.
    """
    try:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_loader, val_loader)
        suggestion = lr_finder.suggestion()
        logger.info(f"Suggested LR: {suggestion:.2e}")
        return suggestion

    except Exception:
        logger.exception("Learning rate finder failed")
        raise

def train(
    trainer: pl.Trainer, 
    model: TemporalFusionTransformer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> str:
    """
    Fit the model and return the best checkpoint path.
    """
    try:
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = trainer.callbacks[-1].best_model_path
        logger.info(f"Best checkpoint -> {ckpt_path}")
        return ckpt_path
    except Exception:
        logger.exception("Error while training TFT")
        raise

def evaluate(
    ckpt_path: str,
    val_loader: torch.utils.data.DataLoader,
    plot_date: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load best model and predict values
    """
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        # actuals = torch.cat([y[0] for x, y in iter(val_loader)].to('cpu'))
        # predictions = model.predict(val_dataloader, trainer_kwargs=dict(accelerator='auto'))
        raw_prediction = model.predict(val_loader, mode='raw', return_x=True)
        print(raw_prediction.output)
        model.plot_prediction(raw_prediction.x, raw_prediction.output, idx=0)
        plt.show()


        preds = model.predict(val_loader)
        targets = torch.cat([y[0] for _, y in val_loader]).to(preds.device)

        mae = MAE().to(preds.device)
        smape = SMAPE().to(preds.device)

        logger.info(f"Validation MAE : {mae(preds, targets):.3f}")
        logger.info(f"Validation SMAPE : {smape(preds, targets):.3f} %")



        return preds, targets
    except Exception:
        logger.exception("Evaluation failed")
        raise

def save_predictions(
    preds: torch.Tensor,
    output_path: Path,
    prediction_length: int,
) -> None:
    """
    Save predictions to csv.
    """
    try:
        df_pred = pd.DataFrame(
            preds.cpu().numpy(),
            columns = [f"t+{i}" for i in range(1, prediction_length + 1)]
        )
        df_pred.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    except Exception:
        logger.exception("Failed to save predictions.")
        raise

def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    df = load_data(args.csv_path)
    df = feature_processing(df)
    
    training_dataset, validation_dataset = create_datasets(
        df, args.max_encoder_length, args.max_prediction_length 
    )

    train_loader, val_loader = get_dataloaders(
        training_dataset, validation_dataset, args.batch_size
    )

    # quick persistence baseline
    baseline = Baseline().predict(val_loader, return_y=True)
    logger.info(f"Persistence MAE: {MAE()(baseline.output, baseline.y):.3f}")

    callbacks = [
        # EarlyStopping(monitor="val_loss", min_delta=1e-7, patience=10, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor="val_loss",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min"
        )
    ]

    tb_logger = TensorBoardLogger("lightning_logs", name="tft_run")

    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=args.max_epochs,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=tb_logger,
    )

    model = build_model(training_dataset)

    # Training model 
    if args.train:
        # model.hparams.learning_rate = find_learning_rate(
        #     trainer, model, train_loader, val_loader
        # )
        ckpt = train(trainer, model, train_loader, val_loader)
        preds, targets = evaluate(ckpt, val_loader)
        print(preds, "\n", targets)
        save_predictions(preds, Path("output/val_predictions.csv"), args.max_prediction_length)
    

    # loading model to plot forecasts
    if DEBUG:
        # irradiance model
        # ckpt = "lightning_logs/tft_run/version_13/checkpoints/tft-epoch=43-val_loss=3.6114.ckpt"
        # temperature model
        ckpt = "models/irradiance_model.ckpt"
        preds, targets = evaluate(ckpt, val_loader, args.plot_date)

        print(preds, "\n", targets)
        # convert to NumPy
        preds_np = preds.detach().cpu().numpy() - 273.15
        targets_np = targets.detach().cpu().numpy() - 273.15
        print(preds_np)
        print(targets_np[0])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Script terminated with errors")
        sys.exit(1)
