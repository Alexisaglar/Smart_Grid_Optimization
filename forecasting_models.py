
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

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
TRAIN_TFT_MODEL = False
TRAIN_LSTM_MODEL = False

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and compare forecasting models")
    parser.add_argument("--csv_path", type=Path, default=CSV_PATH)
    parser.add_argument("--predicting", type=str, default='GHI', help="Target variable: 'GHI' or 't2m'")
    # ### CHANGE ### Restored days and simulation_hours arguments for flexibility
    parser.add_argument("--days", type=int, default=0, help="Start day offset into the validation year (2024).")
    parser.add_argument("--simulation_hours", type=int, default=168, help="Number of hours to simulate.")
    parser.add_argument("--max_encoder_length", type=int, default=MAX_ENCODER_LENGTH)
    parser.add_argument("--max_prediction_length", type=int, default=MAX_PREDICTION_LENGTH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument(
        "--train_tft", action="store_true", default=TRAIN_TFT_MODEL,
        help="If set, run TFT model training."
    )
    parser.add_argument(
        "--train_lstm", action="store_true", default=TRAIN_LSTM_MODEL,
        help="If set, run LSTM model training."
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
        
        hour = df['datetime'].dt.hour
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['wind_dir'] = np.degrees(np.arctan2(df['v10'], df['u10'])) % 360
        df['sin_wdir'] = np.sin(np.deg2rad(df['wind_dir']))
        df['cos_wdir'] = np.cos(np.deg2rad(df['wind_dir']))

        df['t2m'] = df['t2m'] + 273.15
        df['group_id'] = 'site0-NCL'
        df['group_id'] = df['group_id'].astype('category')

        df = df.sort_values(['group_id', 'time_idx'])

        ### NEW ### Add lagged features for the target variables
        df = df.sort_values('time_idx').reset_index(drop=True)
        for lag in [24, 48, 168]:
            df[f'GHI_lag_{lag}'] = df['GHI'].shift(lag)
            df[f't2m_lag_{lag}'] = df['t2m'].shift(lag)
        # df['t2m_lag24'] = df.groupby('group_id', observed=True)['t2m'].shift(24)
        df = df.dropna().reset_index(drop=True)
        df['time_idx'] = df.index

    except Exception:
        logger.exception("Error while processing features")
        raise
    return df

# ---------------- TFT Functions ----------------

def create_tft_datasets(
    df: pd.DataFrame, max_encoder_length: int, max_prediction_length: int, predicting: str
) -> tuple[TimeSeriesDataSet, pd.DataFrame]:
    """Creates training and validation datasets for TFT."""
    validation_cutoff = df['time_idx'].max() - (365 * 24)
    training_df = df[df.time_idx <= validation_cutoff]
    validation_df = df[df.time_idx > validation_cutoff]

    logger.info(f"TFT Training data from {training_df.datetime.min()} to {training_df.datetime.max()}")
    logger.info(f"TFT Validation data from {validation_df.datetime.min()} to {validation_df.datetime.max()}")

    time_varying_known_categoricals = ['day_of_week', 'day_of_month', 'month', 'hour']
    time_varying_known_reals = ['time_idx', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
    time_varying_unknown_reals = list(df.select_dtypes(include=np.number).columns)
    
    for col in [predicting, 'time_idx'] + time_varying_known_reals:
        if col in time_varying_unknown_reals:
            time_varying_unknown_reals.remove(col)
    
    training_dataset = TimeSeriesDataSet(
        training_df, time_idx='time_idx', target=predicting, group_ids=['group_id'],
        max_encoder_length=max_encoder_length, max_prediction_length=max_prediction_length,
        static_categoricals=[], static_reals=[],
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    return training_dataset, validation_df


def train_tft_model(
    training_dataset: TimeSeriesDataSet, validation_df: pd.DataFrame, args: argparse.Namespace
) -> str:
    """Trains the TFT model."""
    val_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_df)
    train_loader = training_dataset.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=args.batch_size * 10, num_workers=0)

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"),
        LearningRateMonitor(),
        ModelCheckpoint(monitor="val_loss", filename="tft-{epoch:02d}-{val_loss:.4f}", save_top_k=1, mode="min")
    ]
    tb_logger = TensorBoardLogger("lightning_logs", name=f"tft_{training_dataset.target}_run")
    trainer = pl.Trainer(
        accelerator='auto', max_epochs=args.max_epochs, gradient_clip_val=0.1,
        callbacks=callbacks, logger=tb_logger,
    )
    
    model = TemporalFusionTransformer.from_dataset(
        training_dataset, learning_rate=0.03, hidden_size=64, attention_head_size=4,
        dropout=0.2, hidden_continuous_size=128, loss=QuantileLoss(),
        log_interval=10, optimizer='ranger',
    )
    
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader, val_loader)
    model.hparams.learning_rate = lr_finder.suggestion()
    logger.info(f"Using suggested learning rate for TFT: {model.hparams.learning_rate:.2e}")

    trainer.fit(model, train_loader, val_loader)
    return trainer.checkpoint_callback.best_model_path


# ---------------- LSTM Functions ----------------

class LSTMForecastModel(nn.Module):
    """An advanced LSTM model with multiple layers and dropout."""
    def __init__(self, input_size=1, hidden_layer_size=128, num_layers=1, output_size=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        return predictions


def create_lstm_sequences(features, target, sequence_length):
    """
    Creates sequences for LSTM training from separate feature and target arrays.
    This prevents data leakage.
    """
    X, y = [], []
    # Iterate through the data to create sequences of length `sequence_length`
    for i in range(len(features) - sequence_length):
        # The input sequence (X) is a slice of the features
        X.append(features[i:(i + sequence_length)])
        # The target (y) is the single value from the target array that immediately follows the input sequence
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def train_lstm_model(df, predicting, max_encoder_length, args):
    """
    Trains the LSTM model with a validation loop and early stopping.
    This version is corrected to prevent data leakage.
    """
    logger.info(f"--- Starting LSTM Model Training for {predicting} ---")

    # Define the features (X) and the target (y)
    features_x = ['wind_speed', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
    feature_y = [predicting]

    # Split data into training and validation sets
    end_of_training = df['time_idx'].max() - (365 * 24)
    end_of_lstm_train = end_of_training - (365 * 24)

    train_df = df[df.time_idx <= end_of_lstm_train]
    val_df = df[(df.time_idx > end_of_lstm_train) & (df.time_idx <= end_of_training)]

    # Initialize and fit scalers ONLY on the training data
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(train_df[features_x])
    y_train_scaled = y_scaler.fit_transform(train_df[feature_y])

    # Transform the validation data using the fitted scalers
    x_val_scaled = x_scaler.transform(val_df[features_x])
    y_val_scaled = y_scaler.transform(val_df[feature_y])

    # --- FIX APPLIED HERE ---
    # Create sequences from the separate, scaled feature and target arrays.
    # We no longer combine them, which was the source of the leak.
    # .flatten() is used on y_..._scaled to turn it from a column vector into a simple array.
    X_train, y_train = create_lstm_sequences(x_train_scaled, y_train_scaled.flatten(), max_encoder_length)
    X_val, y_val = create_lstm_sequences(x_val_scaled, y_val_scaled.flatten(), max_encoder_length)
    # --- END OF FIX ---

    # Create PyTorch TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --- MODEL INPUT SIZE IS NOW CORRECT ---
    # The input_size is the number of features in our sequences (e.g., 5 in this case).
    # X_train.shape[2] correctly captures this dimension.
    model = LSTMForecastModel(input_size=X_train.shape[2])
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping
    epochs = 100
    patience = 5
    min_val_loss = float('inf')
    epochs_no_improve = 0
    model_path = f"models/lstm_{predicting}.pt"

    for epoch in range(epochs):
        model.train()
        # Using a variable for running training loss if needed, but last batch loss is often sufficient
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += loss_function(y_pred, y_batch).item()
        
        # Average the losses over the number of batches
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'LSTM Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

        # Early stopping logic
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"Validation loss decreased. Saving model to {model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            logger.info("Early stopping triggered.")
            break
            
    logger.info(f"LSTM training complete. Best model saved to: {model_path}")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(x_scaler, f"models/lstm_x_scaler_{predicting}.pkl")
    joblib.dump(y_scaler, f"models/lstm_y_scaler_{predicting}.pkl")

    # Load the best performing model for returning
    best_model = LSTMForecastModel(input_size=X_train.shape[2])
    best_model.load_state_dict(torch.load(model_path))

    return best_model, x_scaler, y_scaler, features_x, model_path

# ---------------- Evaluation and Plotting ----------------

def plot_results(results_df, maes, predicting, simulation_hours):
    """Creates a high-quality figure comparing the four forecasting strategies."""
    logger.info(f"Generating publication-quality figure for {simulation_hours}-hour window...")
    plt.style.use("seaborn-v0_8-ticks")
    unit = "(W/m²)" if predicting == "GHI" else "(°C)"
    title_predicting = 'Global Horizontal Irradiance' if predicting == 'GHI' else "Temperature"
    ylim = 1000 if predicting == 'GHI' else 25
    plt.rcParams.update({
        'font.size': 35, 'axes.labelsize': 30, 'axes.titlesize': 30,
        'xtick.labelsize': 30, 'ytick.labelsize': 30, 'legend.fontsize': 30,
    })
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_df = results_df.head(simulation_hours)
    plot_df.to_csv('4_days_december_t2m.csv')
    
    ax.fill_between(plot_df.index, plot_df['rolling_pred_p10'], plot_df['rolling_pred_p90'], color='#56B4E9', alpha=0.3, label='TFT Rolling 10th-90th Percentile')
    ax.plot(plot_df.index, plot_df['rolling_pred_p50'], label=f'TFT Rolling', color='#0072B2', linewidth=2, zorder=6)
    ax.plot(plot_df.index, plot_df['day_ahead_pred'], label=f'TFT Day-Ahead', color='#D55E00', linestyle='--', linewidth=2, zorder=4)
    ax.plot(plot_df.index, plot_df['actual'], label=f'Actual', color='black', linewidth=2.5, zorder=5)
    ax.plot(plot_df.index, plot_df['naive_pred'], label=f'Naive', color='gray', linestyle=':', linewidth=2, zorder=2)
    ax.plot(plot_df.index, plot_df['lstm_pred'], label=f'LSTM', color='#009E73', linestyle='-.', linewidth=2, zorder=3)
    # ax.set_title(f'Forecasting Strategy Comparison: {title_predicting}')
    ax.set_ylabel(f'{title_predicting} {unit}')
    ax.set_xlabel('Date')
    ax.set_ylim(0, ylim)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(f"full_strategy_comparison_{predicting}.png", dpi=300)
    plt.show()
    plt.rcdefaults()


def run_evaluation(
    tft_model, lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x,
    full_df, validation_df, max_encoder_length, max_prediction_length, predicting,
    days, simulation_hours
):
    """Simulates and evaluates all four forecasting strategies over a specified window."""
    logger.info(f"--- Starting Evaluation for {predicting} ---")
    logger.info(f"Window: {simulation_hours} hours, starting {days} days into the validation set.")
    
    results = []
    start_time_idx = validation_df['time_idx'].min() + (days * 24)
    
    for t in range(simulation_hours):
        current_time_idx = start_time_idx + t
        current_datetime = full_df.loc[full_df.time_idx == current_time_idx, 'datetime'].iloc[0]
        
        if (t + 1) % 24 == 0:
             logger.info(f"Simulating hour {t+1}/{simulation_hours} ({current_datetime})...")
        
        # --- 1. Day-Ahead (Static) TFT Forecast ---
        # ### CHANGE ### Reverted to update every 24 hours for realistic simulation
        if t % 24 == 0:
            day_ahead_input_df = full_df[
                (full_df.time_idx >= current_time_idx - max_encoder_length) &
                (full_df.time_idx < current_time_idx + max_prediction_length)
            ]
            day_ahead_full_forecast = tft_model.predict(day_ahead_input_df, mode="quantiles", return_x=False)
        day_ahead_pred = day_ahead_full_forecast[0, t % 24, 3].item()

        # --- 2. Rolling Horizon (Adaptive) TFT Forecast ---
        rolling_input_df = full_df[full_df.time_idx.between(current_time_idx - max_encoder_length, current_time_idx + max_prediction_length -1)]
        rolling_full_forecast = tft_model.predict(rolling_input_df, mode="quantiles", return_x=False)
        rolling_p10, rolling_p50, rolling_p90 = [rolling_full_forecast[0, 0, i].item() for i in [0, 3, 6]]

        # --- 3. LSTM Forecast ---
        encoder_start = current_time_idx - max_encoder_length
        encoder_end = current_time_idx - 1

        # You only need the input features (X) to make a prediction
        lstm_input_df_x = full_df[full_df.time_idx.between(encoder_start, encoder_end)][lstm_features_x]

        # Scale the features
        lstm_input_scaled_x = lstm_x_scaler.transform(lstm_input_df_x)

        # FIX APPLIED HERE: Create the tensor ONLY from the scaled features (5 columns)
        # Do not concatenate the target variable.
        lstm_input_tensor = torch.tensor(lstm_input_scaled_x, dtype=torch.float32).unsqueeze(0)

        # The rest of the code works as before
        lstm_model.eval()
        with torch.no_grad():
            # This will now work, as the model expects 5 features and the tensor has 5 features
            lstm_pred_scaled = lstm_model(lstm_input_tensor)

        lstm_pred = lstm_y_scaler.inverse_transform(lstm_pred_scaled.numpy())[0, 0]
        # --- 4. Naive (Persistence) Forecast ---
        naive_pred = full_df.loc[full_df.time_idx == current_time_idx - 24, predicting].iloc[0]

        # --- 5. Get Actual Value & Store ---
        actual = full_df.loc[full_df.time_idx == current_time_idx, predicting].iloc[0]
        
        res = {'timestamp': current_datetime, 'actual': actual, 'day_ahead_pred': day_ahead_pred,
               'rolling_pred_p10': rolling_p10, 'rolling_pred_p50': rolling_p50, 'rolling_pred_p90': rolling_p90,
               'lstm_pred': lstm_pred, 'naive_pred': naive_pred}

        if predicting == "t2m":
            for key in res:
                if key != 'timestamp' and res[key] is not np.nan: res[key] -= 273.15
        
        results.append(res)

    # --- Calculate Final Metrics ---
    results_df = pd.DataFrame(results).set_index('timestamp').dropna()
    
    models_to_evaluate = {
        "Naive (Persistence)": "naive_pred", "LSTM": "lstm_pred",
        "TFT Day-Ahead": "day_ahead_pred", "TFT Rolling": "rolling_pred_p50"
    }
    
    final_metrics = {}
    for name, col in models_to_evaluate.items():
        actuals, preds = results_df['actual'], results_df[col]
        final_metrics[name] = {
            "MAE": mean_absolute_error(actuals, preds),
            "RMSE": mean_squared_error(actuals, preds, squared=False)
        }

    logger.info(f"--- Evaluation Results for {predicting} ---")
    print("-" * 60)
    print(f"Metrics for {simulation_hours} hours starting on day {days} of validation set")
    print("-" * 60)
    print(f"{'Model':<25} | {'MAE':<15} | {'RMSE':<15}")
    print("-" * 60)
    for name, metrics in final_metrics.items():
        print(f"{name:<25} | {metrics['MAE']:<15.2f} | {metrics['RMSE']:<15.2f}")
    print("-" * 60)
    
    return results_df, {name: metrics['MAE'] for name, metrics in final_metrics.items()}


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    df = load_data(args.csv_path)
    df = feature_processing(df)

    # --- TFT Model ---
    tft_training_dataset, validation_df = create_tft_datasets(
        df, args.max_encoder_length, args.max_prediction_length, args.predicting
    )
    if args.train_tft:
        logger.info(f"--- Starting TFT Model Training for {args.predicting} ---")
        # new GHI: lightning_logs/tft_GHI_run/version_1/checkpoints/tft-epoch=20-val_loss=21.6691.ckpt
        # new_GHI: lightning_logs/tft_GHI_run/version_3/checkpoints/tft-epoch=38-val_loss=20.9468.ckpt
        # new_t2m: lightning_logs/tft_t2m_run/version_1/checkpoints/tft-epoch=47-val_loss=0.9470.ckpt
        best_tft_ckpt = train_tft_model(tft_training_dataset, validation_df, args)
    else:
        best_tft_ckpt = f"models/new_tft_{'irradiance' if args.predicting == 'GHI' else 'temperature'}.ckpt"
    logger.info(f"Using TFT model: {best_tft_ckpt}")
    
    # --- LSTM Model ---
    if args.train_lstm:
        lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x, _ = train_lstm_model(df, args.predicting, args.max_encoder_length, args)
    else:
        try:
            lstm_model_path = f"models/lstm_{args.predicting}.pt"
            logger.info(f"Skipping LSTM training. Loading existing model and scalers: {lstm_model_path}")
            lstm_x_scaler = joblib.load(f"models/lstm_x_scaler_{args.predicting}.pkl")
            lstm_y_scaler = joblib.load(f"models/lstm_y_scaler_{args.predicting}.pkl")
            lstm_features_x = ['wind_speed', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
            
            # --- FIX APPLIED HERE ---
            # The input size is simply the number of features.
            input_size = len(lstm_features_x) # This now correctly calculates 5
            
            # This now creates a model expecting 5 features...
            lstm_model = LSTMForecastModel(input_size=input_size) 
            # ...which perfectly matches the 5-feature model saved on disk.
            lstm_model.load_state_dict(torch.load(lstm_model_path))
        except FileNotFoundError:
            logger.error("LSTM model or scaler not found. Please train first using --train_lstm flag.")
            sys.exit(1)

    # --- Run Evaluation and Plotting ---
    try:
        tft_model = TemporalFusionTransformer.load_from_checkpoint(best_tft_ckpt)
        
        results_df, maes = run_evaluation(
            tft_model=tft_model, lstm_model=lstm_model,
            lstm_x_scaler=lstm_x_scaler, lstm_y_scaler=lstm_y_scaler,
            lstm_features_x=lstm_features_x, full_df=df, validation_df=validation_df,
            max_encoder_length=args.max_encoder_length, max_prediction_length=args.max_prediction_length,
            predicting=args.predicting, days=args.days, simulation_hours=args.simulation_hours
        )
        
        plot_results(results_df, maes, args.predicting, args.simulation_hours)
        
    except FileNotFoundError as e:
        logger.error(f"A model file was not found: {e}. Please train the models first or correct the path.")
        sys.exit(1)
    except Exception:
        logger.exception("An error occurred during the comparison simulation.")

if __name__ == '__main__':
    main()
