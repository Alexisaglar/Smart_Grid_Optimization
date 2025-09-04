import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_single_forecast_plot(df, palette, line_styles, y_label, figure_title, output_filename):
    """
    Creates a single, full-figure plot for a 4-day forecast comparison.

    Args:
        df (pd.DataFrame): The 4-day dataframe for plotting.
        palette (dict): A dictionary mapping model names to colors.
        line_styles (dict): A dictionary mapping model names to line styles.
        y_label (str): The label for the y-axis.
        figure_title (str): The main title for the figure.
        output_filename (str): The name of the file to save the plot as.
    """
    # --- 1. Setup Plot Aesthetics ---
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    # Figure size for a single, comprehensive plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), dpi=150)
    
    # --- FONT SIZE ADJUSTMENTS ---
    TITLE_FONT_SIZE = 35
    LABEL_FONT_SIZE = 30
    LEGEND_FONT_SIZE = 28
    TICK_FONT_SIZE = 30
    
    # --- 2. Plot All Data on the Single Axis ---
    # ax.set_title(figure_title, fontsize=TITLE_FONT_SIZE, pad=20)
    
    ax.fill_between(df.index, df['rolling_pred_p10'], df['rolling_pred_p90'], 
                     color=palette['TFT 10th-90th Percentile'], alpha=0.15, label='TFT 10th-90th Percentile')
    ax.plot(df.index, df['actual'], label='Actual', color=palette['Actual'], linestyle=line_styles['Actual'], linewidth=3.5)
    ax.plot(df.index, df['rolling_pred_p50'], label='TFT Rolling Horizon', color=palette['TFT Rolling Horizon'], linestyle=line_styles['TFT Rolling Horizon'], linewidth=3)
    ax.plot(df.index, df['day_ahead_pred'], label='TFT Day-Ahead', color=palette['TFT Day-Ahead'], linestyle=line_styles['TFT Day-Ahead'], linewidth=3)
    ax.plot(df.index, df['lstm_pred'], label='LSTM', color=palette['LSTM'], linestyle=line_styles['LSTM'], linewidth=3)
    ax.plot(df.index, df['naive_pred'], label='Naive', color=palette['Naive'], linestyle=line_styles['Naive'], linewidth=2.5)

    # --- 3. Customize Axes and Labels for 4 Days ---
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel('Time', fontsize=LABEL_FONT_SIZE)
    
    # Set ticks for the center of each of the 4 days (assuming 24h data)
    ax.set_xticks([11, 35, 59, 83])
    ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4'], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Set black border
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_edgecolor('black')
        ax.spines[spine].set_linewidth(1.5)

    # --- 4. Create Integrated Legend ---
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 3, 4, 5, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], 
               loc='upper left', ncol=3, fontsize=LEGEND_FONT_SIZE, 
               frameon=True, facecolor='white', framealpha=0.8)

    # --- 5. Adjust Layout and Save ---
    fig.tight_layout()
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

    # --- List of plots to generate ---
    # Update these filenames to your new 4-day data files
    plot_configurations = [
        {
            "season": "Winter",
            "variable": "GHI",
            "file": "4_days_december_ghi.csv",
            "ylabel": rf"GHI (W/m$^2$)"
        },
        {
            "season": "Summer",
            "variable": "GHI",
            "file": "4_days_july_ghi.csv",
            "ylabel": rf"GHI (W/m$^2$)"
        },
        {
            "season": "Winter",
            "variable": "Temperature",
            "file": "4_days_december_t2m.csv",
            "ylabel": rf"Temperature ($^\circ$C)"
        },
        {
            "season": "Summer",
            "variable": "Temperature",
            "file": "4_days_july_t2m.csv",
            "ylabel": rf"Temperature ($^\circ$C)"
        }
    ]

    # --- Loop through configurations and generate plots ---
    for config in plot_configurations:
        try:
            print(f"--- Generating plot for {config['season']} {config['variable']} ---")
            # Load the 4-day data directly
            four_day_df = pd.read_csv(config['file'])
            
            figure_title = f"{config['season']} {config['variable']} Forecast (4-Day Period)"
            output_filename = f"{config['season'].lower()}_{config['variable'].lower()}_4day_forecast.png"

            create_single_forecast_plot(
                df=four_day_df,
                palette=palette,
                line_styles=line_styles,
                y_label=config['ylabel'],
                figure_title=figure_title,
                output_filename=output_filename
            )

        except FileNotFoundError:
            print(f"Warning: Could not find file '{config['file']}'. Please update the filenames. Skipping this plot.")
        except Exception as e:
            print(f"An error occurred while generating the plot for {config['file']}: {e}")


