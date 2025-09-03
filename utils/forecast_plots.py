import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_comparison(winter_df, summer_df, palette, line_styles, y_label, figure_title, output_filename):
    """
    Creates, customizes, and saves a side-by-side forecast comparison plot with a
    compact layout to minimize space between the subplots.

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), sharey=True, dpi=150)
    
    # --- FONT SIZE ADJUSTMENTS ---
    TITLE_FONT_SIZE = 35
    SUBTITLE_FONT_SIZE = 32
    LABEL_FONT_SIZE = 30
    LEGEND_FONT_SIZE = 28
    TICK_FONT_SIZE = 30
    
    # --- 2. Plot Winter Data (Left Side) ---
    ax1.set_title('Winter', fontsize=SUBTITLE_FONT_SIZE)
    ax1.fill_between(winter_df.index, winter_df['rolling_pred_p10'], winter_df['rolling_pred_p90'], 
                     color=palette['TFT 10th-90th Percentile'], alpha=0.15, label='TFT 10th-90th Percentile')
    ax1.plot(winter_df.index, winter_df['actual'], label='Actual', color=palette['Actual'], linestyle=line_styles['Actual'], linewidth=3)
    ax1.plot(winter_df.index, winter_df['rolling_pred_p50'], label='TFT Rolling Horizon', color=palette['TFT Rolling Horizon'], linestyle=line_styles['TFT Rolling Horizon'], linewidth=2.5)
    ax1.plot(winter_df.index, winter_df['day_ahead_pred'], label='TFT Day-Ahead', color=palette['TFT Day-Ahead'], linestyle=line_styles['TFT Day-Ahead'], linewidth=2.5)
    ax1.plot(winter_df.index, winter_df['lstm_pred'], label='LSTM', color=palette['LSTM'], linestyle=line_styles['LSTM'], linewidth=2.5)
    ax1.plot(winter_df.index, winter_df['naive_pred'], label='Naive', color=palette['Naive'], linestyle=line_styles['Naive'], linewidth=2)

    # --- 3. Plot Summer Data (Right Side) ---
    ax2.set_title('Summer', fontsize=SUBTITLE_FONT_SIZE)
    ax2.fill_between(summer_df.index, summer_df['rolling_pred_p10'], summer_df['rolling_pred_p90'], 
                     color=palette['TFT 10th-90th Percentile'], alpha=0.15)
    ax2.plot(summer_df.index, summer_df['actual'], color=palette['Actual'], linestyle=line_styles['Actual'], linewidth=3)
    ax2.plot(summer_df.index, summer_df['rolling_pred_p50'], color=palette['TFT Rolling Horizon'], linestyle=line_styles['TFT Rolling Horizon'], linewidth=2.5)
    ax2.plot(summer_df.index, summer_df['day_ahead_pred'], color=palette['TFT Day-Ahead'], linestyle=line_styles['TFT Day-Ahead'], linewidth=2.5)
    ax2.plot(summer_df.index, summer_df['lstm_pred'], color=palette['LSTM'], linestyle=line_styles['LSTM'], linewidth=2.5)
    ax2.plot(summer_df.index, summer_df['naive_pred'], color=palette['Naive'], linestyle=line_styles['Naive'], linewidth=2)

    # --- 4. Customize Axes and Labels ---
    ax1.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Time', fontsize=LABEL_FONT_SIZE)
        ax.set_xticks([11, 35])
        ax.set_xticklabels(['Day 1', 'Day 2'], fontsize=TICK_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_edgecolor('black')
            ax.spines[spine].set_linewidth(1.5)

    # --- 5. Create Integrated Legend ---
    handles, labels = ax1.get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]
    ax1.legend([handles[i] for i in order], [labels[i] for i in order], 
               loc='upper left', ncol=1, fontsize=LEGEND_FONT_SIZE, 
               frameon=True, facecolor='white', framealpha=0.7)

    # --- 6. Adjust Layout and Save ---
    # Manually reduce the width space (wspace) between the two subplots
    plt.subplots_adjust(wspace=0.05)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{output_filename}'")
    plt.show()

if __name__ == '__main__':
    # --- Define Shared Palette and Styles ---
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
            winter_df=df_winter_ghi, summer_df=df_summer_ghi, palette=palette,
            line_styles=line_styles, y_label='GHI (W/m²)',
            figure_title='Multi-Model GHI Forecast Comparison',
            output_filename='ghi_forecast_comparison_compact.png'
        )
    except FileNotFoundError as e:
        print(f"Could not generate GHI plot. Error: {e}")

    # --- Generate Temperature Plot ---
    try:
        df_winter_temp = pd.read_csv('2_days_january_temperature.csv')
        df_summer_temp = pd.read_csv('2_days_july_temperature.csv')
        plot_forecast_comparison(
            winter_df=df_winter_temp, summer_df=df_summer_temp, palette=palette,
            line_styles=line_styles, y_label='Temperature (°C)',
            figure_title='Multi-Model Temperature Forecast Comparison',
            output_filename='temperature_forecast_comparison_compact.png'
        )
    except FileNotFoundError as e:
        print(f"Could not generate Temperature plot. Error: {e}")

