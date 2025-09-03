import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_comparison(winter_df, summer_df, palette, line_styles, y_label, figure_title, output_filename):
    """
    Creates, customizes, and saves a side-by-side winter vs. summer forecast comparison plot for a single variable.

    Args:
        winter_df (pd.DataFrame): The dataframe for winter data.
        summer_df (pd.DataFrame): The dataframe for summer data.
        palette (dict): A dictionary mapping model names to colors.
        line_styles (dict): A dictionary mapping model names to line styles.
        y_label (str): The label for the y-axis.
        figure_title (str): The main title for the entire figure.
        output_filename (str): The name of the file to save the plot as.
    """
    # --- 1. Setup Plot Aesthetics ---
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True, dpi=150)
    # fig.suptitle(figure_title, fontsize=22, y=1.0)

    # --- 2. Plot Winter Data (Left Side) ---
    ax1.set_title('Winter', fontsize=18)
    ax1.fill_between(winter_df.index, winter_df['rolling_pred_p10'], winter_df['rolling_pred_p90'], 
                     color=palette['TFT 10th-90th Percentile'], alpha=0.15, label='TFT 10th-90th Percentile')
    ax1.plot(winter_df.index, winter_df['actual'], label='Actual', color=palette['Actual'], linestyle=line_styles['Actual'], linewidth=2.5)
    ax1.plot(winter_df.index, winter_df['rolling_pred_p50'], label='TFT Rolling Horizon', color=palette['TFT Rolling Horizon'], linestyle=line_styles['TFT Rolling Horizon'], linewidth=2.2)
    ax1.plot(winter_df.index, winter_df['day_ahead_pred'], label='TFT Day-Ahead', color=palette['TFT Day-Ahead'], linestyle=line_styles['TFT Day-Ahead'], linewidth=2)
    ax1.plot(winter_df.index, winter_df['lstm_pred'], label='LSTM', color=palette['LSTM'], linestyle=line_styles['LSTM'], linewidth=2)
    ax1.plot(winter_df.index, winter_df['naive_pred'], label='Naive', color=palette['Naive'], linestyle=line_styles['Naive'], linewidth=1.5)

    # --- 3. Plot Summer Data (Right Side) ---
    ax2.set_title('Summer', fontsize=18)
    ax2.fill_between(summer_df.index, summer_df['rolling_pred_p10'], summer_df['rolling_pred_p90'], 
                     color=palette['TFT 10th-90th Percentile'], alpha=0.15)
    ax2.plot(summer_df.index, summer_df['actual'], color=palette['Actual'], linestyle=line_styles['Actual'], linewidth=2.5)
    ax2.plot(summer_df.index, summer_df['rolling_pred_p50'], color=palette['TFT Rolling Horizon'], linestyle=line_styles['TFT Rolling Horizon'], linewidth=2.2)
    ax2.plot(summer_df.index, summer_df['day_ahead_pred'], color=palette['TFT Day-Ahead'], linestyle=line_styles['TFT Day-Ahead'], linewidth=2)
    ax2.plot(summer_df.index, summer_df['lstm_pred'], color=palette['LSTM'], linestyle=line_styles['LSTM'], linewidth=2)
    ax2.plot(summer_df.index, summer_df['naive_pred'], color=palette['Naive'], linestyle=line_styles['Naive'], linewidth=1.5)

    # --- 4. Customize Axes and Labels ---
    ax1.set_ylabel(y_label, fontsize=15)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time', fontsize=15)
        ax.set_xticks([11, 35])
        ax.set_xticklabels(['Day 1', 'Day 2'], fontsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(axis='x', linestyle=':', alpha=0.6)

    # --- 5. Create a Shared Legend at the Top ---
    handles, labels = ax1.get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], 
               loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=6, fontsize=14, frameon=False)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    # --- Define Color Palette and Line Styles (Shared) ---
    palette = {
        'Actual': '#003f5c',
        'TFT Rolling Horizon': '#d62728',
        'TFT Day-Ahead': '#ff7f0e',
        'LSTM': '#58508d',
        'Naive': '#bc5090',
        'TFT 10th-90th Percentile': '#ff7f0e' 
    }
    line_styles = {
        'Actual': '-',
        'TFT Rolling Horizon': '-',
        'TFT Day-Ahead': '--',
        'LSTM': ':',
        'Naive': '-.'
    }

    # --- Generate GHI Plot ---
    try:
        df_winter_ghi = pd.read_csv('2_days_january_ghi.csv')
        df_summer_ghi = pd.read_csv('2_days_july_ghi.csv')
        
        plot_forecast_comparison(
            winter_df=df_winter_ghi,
            summer_df=df_summer_ghi,
            palette=palette,
            line_styles=line_styles,
            y_label='Global Horizontal Irradiance (W/m²)',
            figure_title='Multi-Model GHI Forecast Comparison',
            output_filename='ghi_forecast_comparison.png'
        )
    except FileNotFoundError as e:
        print(f"Could not generate GHI plot. Error: {e}")

    # --- Generate Temperature Plot ---
    try:
        df_winter_temp = pd.read_csv('2_days_january_temperature.csv')
        df_summer_temp = pd.read_csv('2_days_july_temperature.csv')

        plot_forecast_comparison(
            winter_df=df_winter_temp,
            summer_df=df_summer_temp,
            palette=palette,
            line_styles=line_styles,
            y_label='Temperature (°C)',
            figure_title='Multi-Model Temperature Forecast Comparison',
            output_filename='temperature_forecast_comparison.png'
        )
    except FileNotFoundError as e:
        print(f"Could not generate Temperature plot. Error: {e}")

