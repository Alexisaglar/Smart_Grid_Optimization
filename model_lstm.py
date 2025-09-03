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
from tqdm import tqdm

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

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
    parser.add_argument("--predicting", type=str, default='GHI', help="Target variable for WINDOW analysis: 'GHI' or 't2m'")
    # Arguments for windowed analysis and plotting
    parser.add_argument("--days", type=int, default=168, help="Start day offset for WINDOW analysis.")
    parser.add_argument("--simulation_hours", type=int, default=168, help="Number of hours to simulate for WINDOW analysis.")
    # New flag for full-year evaluation
    parser.add_argument("--full_year", action="store_true", help="If set, run evaluation on the entire validation year.")
    
    parser.add_argument("--max_encoder_length", type=int, default=MAX_ENCODER_LENGTH)
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
    df['t2m'] = df['t2m'] + 273.15
    df['group_id'] = 'main'
    df = df.sort_values(['group_id', 'time_idx'])
    df = df.drop(columns=['u10', 'v10'])
    for lag in [24, 48, 168]:
        df[f'GHI_lag_{lag}'] = df['GHI'].shift(lag)
        df[f't2m_lag_{lag}'] = df['t2m'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    df['time_idx'] = df.index
    return df

# ---------------- TFT Functions ----------------
def create_tft_datasets(
    df: pd.DataFrame, max_encoder_length: int, predicting: str
) -> tuple[TimeSeriesDataSet, pd.DataFrame]:
    """Creates training and validation datasets for TFT."""
    validation_cutoff = df['time_idx'].max() - (365 * 24)
    training_df = df[df.time_idx <= validation_cutoff]
    validation_df = df[df.time_idx > validation_cutoff]

    time_varying_known_categoricals = ['day_of_week', 'day_of_month', 'month', 'hour']
    time_varying_known_reals = ['time_idx', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
    time_varying_unknown_reals = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in time_varying_known_reals and c != predicting and c != 'time_idx'
    ]
    
    training_dataset = TimeSeriesDataSet(
        training_df, time_idx='time_idx', target=predicting, group_ids=['group_id'],
        max_encoder_length=max_encoder_length, 
        max_prediction_length=MAX_PREDICTION_LENGTH,
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
    return training_dataset, validation_df

def train_tft_model(
    training_dataset: TimeSeriesDataSet, validation_df: pd.DataFrame, args: argparse.Namespace
) -> str:
    """Trains the TFT model."""
    val_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_df, stop_randomization=True)
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
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

def create_lstm_sequences(features, target, sequence_length):
    """Creates sequences for LSTM training from separate feature and target arrays."""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def train_lstm_model(df, predicting, max_encoder_length, args):
    """Trains the LSTM model with a validation loop and early stopping."""
    logger.info(f"--- Starting LSTM Model Training for {predicting} ---")
    features_x = ['wind_speed', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
    feature_y = [predicting]
    end_of_training = df['time_idx'].max() - (365 * 24)
    end_of_lstm_train = end_of_training - (365 * 24)
    train_df = df[df.time_idx <= end_of_lstm_train]
    val_df = df[(df.time_idx > end_of_lstm_train) & (df.time_idx <= end_of_training)]
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(train_df[features_x])
    y_train_scaled = y_scaler.fit_transform(train_df[feature_y])
    x_val_scaled = x_scaler.transform(val_df[features_x])
    y_val_scaled = y_scaler.transform(val_df[feature_y])
    X_train, y_train = create_lstm_sequences(x_train_scaled, y_train_scaled.flatten(), max_encoder_length)
    X_val, y_val = create_lstm_sequences(x_val_scaled, y_val_scaled.flatten(), max_encoder_length)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    model = LSTMForecastModel(input_size=X_train.shape[2])
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    patience = 5
    min_val_loss = float('inf')
    epochs_no_improve = 0
    model_path = f"models/lstm_{predicting}.pt"
    for epoch in range(epochs):
        model.train()
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
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'LSTM Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            logger.info("Early stopping triggered.")
            break
    logger.info(f"LSTM training complete. Best model saved to: {model_path}")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(x_scaler, f"models/lstm_x_scaler_{predicting}.pkl")
    joblib.dump(y_scaler, f"models/lstm_y_scaler_{predicting}.pkl")
    best_model = LSTMForecastModel(input_size=X_train.shape[2])
    best_model.load_state_dict(torch.load(model_path))
    return best_model, x_scaler, y_scaler, features_x, model_path


# ---------------- Evaluation Functions ----------------

## --- FUNCTION 1: For Windowed Analysis & Plotting ---
## --- FUNCTION 1: For Windowed Analysis & Plotting (REVISED) ---
def run_evaluation(
    tft_model, lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x,
    full_df, validation_df, max_encoder_length, predicting,
    days, simulation_hours
):
    """Simulates and evaluates all forecasting strategies over a SPECIFIED WINDOW."""
    logger.info(f"--- Starting WINDOW Evaluation for {predicting} ---")
    logger.info(f"Window: {simulation_hours} hours, starting {days} days into the validation set.")
    
    # --- NEW: Define the full simulation window ---
    start_time_idx = validation_df['time_idx'].min() + (days * 24)
    end_time_idx = start_time_idx + simulation_hours - 1
    
    # --- NEW: Create a single DataFrame for the entire window we need to predict ---
    # We include a buffer for the encoder and future known inputs.
    window_df = full_df[
        full_df.time_idx.between(start_time_idx - max_encoder_length, end_time_idx + MAX_PREDICTION_LENGTH)
    ].copy()

    # --- NEW: Perform one robust, batch prediction for the ROLLING forecast ---
    logger.info("Performing batch prediction for the TFT rolling forecast...")
    rolling_preds_raw = tft_model.predict(
        window_df,
        mode="quantiles",
        return_index=True
    )
    # The model predicts for every possible start time. We store the results in a new DataFrame.
    rolling_preds_df = pd.DataFrame({
        'rolling_pred_p10': rolling_preds_raw.prediction[:, 0, 0], # 10th quantile
        'rolling_pred_p50': rolling_preds_raw.prediction[:, 0, 3], # 50th quantile
        'rolling_pred_p90': rolling_preds_raw.prediction[:, 0, 6], # 90th quantile
    }, index=rolling_preds_raw.index.time_idx)

    # --- NEW: Pre-calculate all DAY-AHEAD forecasts as well ---
    logger.info("Performing batch predictions for the TFT day-ahead forecast...")
    day_ahead_predictions = {}
    # Loop every 24 hours to generate a new day-ahead forecast
    for t_start in range(0, simulation_hours, 24):
        current_time_idx = start_time_idx + t_start
        day_ahead_input_df = full_df[full_df.time_idx.between(current_time_idx - max_encoder_length, current_time_idx + MAX_PREDICTION_LENGTH - 1)]
        forecast = tft_model.predict(day_ahead_input_df, mode="quantiles", return_x=False)
        # Store the full 24-hour forecast in a dictionary for easy lookup
        for i in range(24):
            if (t_start + i) < simulation_hours:
                day_ahead_predictions[current_time_idx + i] = forecast[0, i, 3].item() # P50 prediction

    results = []
    # --- REVISED: The main loop is now a fast and simple assembly loop ---
    # No model prediction happens inside this loop anymore.
    for t in tqdm(range(simulation_hours), desc="Assembling results"):
        current_time_idx = start_time_idx + t
        current_datetime = full_df.loc[full_df.time_idx == current_time_idx, 'datetime'].iloc[0]
        
        # --- REVISED: Look up all pre-calculated TFT predictions ---
        day_ahead_pred = day_ahead_predictions.get(current_time_idx)
        rolling_p10 = rolling_preds_df.loc[current_time_idx, 'rolling_pred_p10']
        rolling_p50 = rolling_preds_df.loc[current_time_idx, 'rolling_pred_p50']
        rolling_p90 = rolling_preds_df.loc[current_time_idx, 'rolling_pred_p90']

        # --- LSTM and Naive predictions (original logic is fine) ---
        encoder_start = current_time_idx - max_encoder_length
        encoder_end = current_time_idx - 1
        lstm_input_df_x = full_df.loc[full_df.time_idx.between(encoder_start, encoder_end), lstm_features_x]
        lstm_input_scaled_x = lstm_x_scaler.transform(lstm_input_df_x)
        lstm_input_tensor = torch.tensor(lstm_input_scaled_x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lstm_pred_scaled = lstm_model(lstm_input_tensor)
        lstm_pred = lstm_y_scaler.inverse_transform(lstm_pred_scaled.numpy())[0, 0]

        naive_pred = full_df.loc[full_df.time_idx == current_time_idx - 24, predicting].iloc[0]
        actual = full_df.loc[full_df.time_idx == current_time_idx, predicting].iloc[0]
        
        res = {'timestamp': current_datetime, 'actual': actual, 'day_ahead_pred': day_ahead_pred,
               'rolling_pred_p10': rolling_p10, 'rolling_pred_p50': rolling_p50, 'rolling_pred_p90': rolling_p90,
               'lstm_pred': lstm_pred, 'naive_pred': naive_pred}

        if predicting == "t2m":
            for key in res:
                if key != 'timestamp' and res[key] is not np.nan: res[key] -= 273.15
        
        results.append(res)

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


## --- FUNCTION 2: For Efficient Full-Year Analysis ---
def evaluate_full_year(
    tft_model, lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x,
    training_dataset, validation_df, max_encoder_length, predicting
):
    """Efficiently evaluates models over the ENTIRE validation dataset using batch prediction."""
    logger.info(f"--- Starting FULL YEAR Evaluation for {predicting} ---")
    
    # 1. Get Actuals and Naive Forecast
    actuals = validation_df[predicting]
    naive_preds = validation_df[predicting].shift(24) # Persistence forecast
    
    # 2. TFT Batch Prediction (fast)
    logger.info("Generating TFT predictions for the full year...")
    
    # --- FIX APPLIED HERE ---
    # Pass the validation DataFrame directly to the predict method.
    # This is more robust as it lets the model handle the data processing,
    # ensuring the feature set matches what it was trained on.
    tft_raw_preds = tft_model.predict(
        validation_df,
        mode="raw",
        return_index=True
    )
    # --- END OF FIX ---

    # The TFT predicts 24 hours ahead for each sample. We only want the first prediction (h=1) for rolling eval.
    tft_preds = tft_raw_preds.prediction[:, 0, 3].numpy() # P50 is the 4th quantile (index 3)
    
    # 3. LSTM Batch Prediction (fast)
    logger.info("Generating LSTM predictions for the full year...")
    val_features_scaled = lstm_x_scaler.transform(validation_df[lstm_features_x])
    val_target_scaled = lstm_y_scaler.transform(validation_df[[predicting]])
    
    X_val, _ = create_lstm_sequences(val_features_scaled, val_target_scaled.flatten(), max_encoder_length)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    lstm_preds_scaled = []
    lstm_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(X_val_tensor), BATCH_SIZE), desc="LSTM Batch Prediction"):
            batch = X_val_tensor[i:i+BATCH_SIZE]
            preds = lstm_model(batch)
            lstm_preds_scaled.append(preds)
    
    lstm_preds_scaled = torch.cat(lstm_preds_scaled).numpy()
    lstm_preds = lstm_y_scaler.inverse_transform(lstm_preds_scaled).flatten()

    # 4. Align all predictions into a single DataFrame
    # Note the slicing to align predictions with actuals
    results = pd.DataFrame({
        'actual': actuals.iloc[max_encoder_length:].values,
        'naive_pred': naive_preds.iloc[max_encoder_length:].values,
        'tft_pred': tft_preds[:len(actuals) - max_encoder_length],
        'lstm_pred': lstm_preds
    })
    
    if predicting == "t2m":
        results -= 273.15 # Convert all columns from Kelvin to Celsius
    
    results.dropna(inplace=True)

    # 5. Calculate Metrics
    final_metrics = {}
    models_to_evaluate = {
        "Naive (Persistence)": "naive_pred", "LSTM": "lstm_pred", "TFT Rolling": "tft_pred"
    }
    for name, col in models_to_evaluate.items():
        final_metrics[name] = {
            "MAE": mean_absolute_error(results['actual'], results[col]),
            "RMSE": mean_squared_error(results['actual'], results[col], squared=False)
        }
    return final_metrics

def plot_results(results_df, maes, predicting, simulation_hours):
    """Creates a high-quality figure comparing the four forecasting strategies."""
    logger.info(f"Generating publication-quality figure for {simulation_hours}-hour window...")
    plt.style.use("seaborn-v0_8-ticks")
    unit = "(W/m²)" if predicting == "GHI" else "(°C)"
    title_predicting = 'Global Horizontal Irradiance' if predicting == 'GHI' else "Temperature"
    plt.rcParams.update({
        'font.size': 20, 'axes.labelsize': 20, 'axes.titlesize': 22,
        'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14,
    })
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_df = results_df.head(simulation_hours)
    
    ax.plot(plot_df.index, plot_df['actual'], label=f'Actual', color='black', linewidth=2.5, zorder=5)
    ax.plot(plot_df.index, plot_df['naive_pred'], label=f'Naive (Persistence) (MAE: {maes["Naive (Persistence)"]:.2f})', color='gray', linestyle=':', linewidth=2, zorder=2)
    ax.plot(plot_df.index, plot_df['lstm_pred'], label=f'LSTM (MAE: {maes["LSTM"]:.2f})', color='#009E73', linestyle='-.', linewidth=2, zorder=3)
    ax.plot(plot_df.index, plot_df['day_ahead_pred'], label=f'TFT Day-Ahead (MAE: {maes["TFT Day-Ahead"]:.2f})', color='#D55E00', linestyle='--', linewidth=2, zorder=4)
    ax.plot(plot_df.index, plot_df['rolling_pred_p50'], label=f'TFT Rolling (MAE: {maes["TFT Rolling"]:.2f})', color='#0072B2', linewidth=2, zorder=6)
    ax.fill_between(plot_df.index, plot_df['rolling_pred_p10'], plot_df['rolling_pred_p90'], color='#56B4E9', alpha=0.3, label='TFT Rolling 10th-90th Percentile')
    ax.set_title(f'Forecasting Strategy Comparison: {title_predicting}')
    ax.set_ylabel(f'{title_predicting} {unit}')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(f"full_strategy_comparison_{predicting}.png", dpi=300)
    plt.show()
    plt.rcdefaults()


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)

    df = load_data(args.csv_path)
    df = feature_processing(df)

    if args.full_year:
        logger.info("MODE: Running full-year evaluation.")
        all_results = {}
        targets_to_evaluate = ['GHI', 't2m']

        for target in targets_to_evaluate:
            logger.info(f"--- Processing Target: {target} ---")
            tft_training_dataset, validation_df = create_tft_datasets(df, args.max_encoder_length, target)
            
            try:
                tft_ckpt = f"models/new_tft_{'irradiance' if target == 'GHI' else 'temperature'}.ckpt"
                tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_ckpt)
            except FileNotFoundError:
                logger.error(f"TFT model not found for {target}. Please train it first.")
                continue

            try:
                lstm_model_path = f"models/lstm_{target}.pt"
                lstm_x_scaler = joblib.load(f"models/lstm_x_scaler_{target}.pkl")
                lstm_y_scaler = joblib.load(f"models/lstm_y_scaler_{target}.pkl")
                lstm_features_x = ['wind_speed', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
                lstm_model = LSTMForecastModel(input_size=len(lstm_features_x))
                lstm_model.load_state_dict(torch.load(lstm_model_path))
            except FileNotFoundError:
                logger.error(f"LSTM model/scaler not found for {target}. Please train it first.")
                continue
            
            metrics = evaluate_full_year(
                tft_model, lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x,
                tft_training_dataset, validation_df, args.max_encoder_length, target
            )
            all_results[target] = metrics
        
        if all_results:
            results_df = pd.DataFrame.from_dict({(i,j): all_results[i][j] 
                                                 for i in all_results.keys() 
                                                 for j in all_results[i].keys()},
                                                orient='index')
            results_df = results_df.unstack(level=0)
            results_df.columns = [f"{col[1].upper()} {col[0]}" for col in results_df.columns]
            print("\n" + "="*80)
            print(" " * 20 + "Final Full-Year Forecasting Performance")
            print("="*80)
            print(results_df.to_string(float_format="%.2f"))
            print("="*80)

    else:
        logger.info("MODE: Running window analysis and plotting.")
        tft_training_dataset, validation_df = create_tft_datasets(df, args.max_encoder_length, args.predicting)
        
        # --- REVISED LOGIC TO TRAIN IF MODEL IS MISSING ---
        Path("models").mkdir(exist_ok=True)
        tft_ckpt = f"models/new_tft_{'irradiance' if args.predicting == 'GHI' else 'temperature'}.ckpt"

        if args.train_tft or not Path(tft_ckpt).exists():
            if args.train_tft:
                logger.info("--- Training new TFT model as requested by --train_tft flag... ---")
            else:
                logger.info(f"--- TFT model not found at {tft_ckpt}. Training new model... ---")
            
            best_model_path = train_tft_model(tft_training_dataset, validation_df, args)
            tft_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        else:
            logger.info(f"--- Loading existing TFT model from {tft_ckpt} ---")
            tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_ckpt)
        # --- END OF REVISED LOGIC ---
            
        try:
            lstm_model_path = f"models/lstm_{args.predicting}.pt"
            lstm_x_scaler = joblib.load(f"models/lstm_x_scaler_{args.predicting}.pkl")
            lstm_y_scaler = joblib.load(f"models/lstm_y_scaler_{args.predicting}.pkl")
            lstm_features_x = ['wind_speed', 'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']
            lstm_model = LSTMForecastModel(input_size=len(lstm_features_x))
            lstm_model.load_state_dict(torch.load(lstm_model_path))
        except FileNotFoundError:
            logger.error(f"LSTM model/scaler not found for {args.predicting}. Please train it first.")
            sys.exit(1)

        results_df, maes = run_evaluation(
            tft_model, lstm_model, lstm_x_scaler, lstm_y_scaler, lstm_features_x,
            df, validation_df, args.max_encoder_length, args.predicting,
            args.days, args.simulation_hours
        )
        
        plot_results(results_df, maes, args.predicting, args.simulation_hours)

if __name__ == '__main__':
    main()
