import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import sys
import os

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

METRICS = [
    ('emissions', 'Emissions', ''),
    ('emissions_rate', 'Emissions Rate', ''),
    ('cpu_power', 'CPU Power', 'W'),
    ('gpu_power', 'GPU Power', 'W'),
    ('ram_power', 'RAM Power', 'W'),
    ('cpu_energy', 'CPU Energy', 'Wh'),
    ('gpu_energy', 'GPU Energy', 'Wh'),
    ('ram_energy', 'RAM Energy', 'Wh'),
    ('energy_consumed', 'Total Energy', 'Wh'),
]


def load_data(file_path):
    """Load CSV and parse timestamp column"""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    if 'timestamp' not in df.columns:
        raise ValueError("CSV is missing 'timestamp' column")
    df = df.sort_values('timestamp')
    return df


def get_run_start_times(run_files):
    """Extract the start timestamp (user_bin==0) from each run stats CSV"""
    run_starts = []
    for rf in run_files:
        if not os.path.exists(rf):
            print(f"Warning: Run stats file '{rf}' not found, skipping.")
            continue
        try:
            stats = pd.read_csv(rf, parse_dates=['bin_start_timestamp'])
            if 'user_bin' in stats.columns:
                first = stats.loc[stats['user_bin'] == 0]
                ts = first['bin_start_timestamp'].iloc[0] if not first.empty else stats['bin_start_timestamp'].iloc[0]
            else:
                ts = stats['bin_start_timestamp'].iloc[0]
            run_starts.append(ts)
        except Exception as e:
            print(f"Warning: Could not parse '{rf}': {e}")
    return run_starts


def plot_metric(df, col, title, unit, run_starts):
    """Plot a single time series with run start markers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'], df[col], linewidth=2, label=title)
    for start in run_starts:
        ax.axvline(start, color='red', linestyle=':', linewidth=1.5)
    ax.set_title(f"{title} Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel('Time')
    ylabel = f"{title} ({unit})" if unit else title
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def create_summary(df, run_starts):
    """Create a summary grid of all metrics with run start markers"""
    available = [(col, title, unit) for col, title, unit in METRICS if col in df.columns]
    n = len(available)
    cols = 3 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    # ensure axes is iterable
    axes = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]

    for ax, (col, title, unit) in zip(axes, available):
        ax.plot(df['timestamp'], df[col], linewidth=2, label=title)
        for start in run_starts:
            ax.axvline(start, color='red', linestyle=':', linewidth=1)
        ax.set_title(title)
        ylabel = f"{unit}" if unit else ''
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.get_xticklabels(), rotation=45)
    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python emissions_plot.py <emissions.csv> [run1.csv run2.csv ...]")
        sys.exit(1)
    emiss_file = sys.argv[1]
    run_files = sys.argv[2:]
    if not os.path.exists(emiss_file):
        print(f"Error: Emissions file '{emiss_file}' not found")
        sys.exit(1)

    df = load_data(emiss_file)
    run_starts = get_run_start_times(run_files) if run_files else []
    print(f"Found {len(run_starts)} run start(s): {run_starts}")

    figs = []
    names = []
    for col, title, unit in METRICS:
        if col not in df.columns:
            continue
        fig = plot_metric(df, col, title, unit, run_starts)
        figs.append(fig)
        names.append(col)

    # summary grid
    summary_fig = create_summary(df, run_starts)
    figs.append(summary_fig)
    names.append('summary')

    # Save all by default
    for name, fig in zip(names, figs):
        filename = f"{name}_time_series.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()

if __name__ == '__main__':
    main()
