import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import seaborn as sns
import sys
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_process_data(file_path):
    """Load and process the system monitoring data"""
    df = pd.read_csv(file_path)

    # Convert timestamp column to datetime (assuming milliseconds)
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['time_str'] = df['datetime'].dt.strftime('%H:%M:%S')
    else:
        raise ValueError("Missing 'timestamp' column in system metrics CSV")

    # CPU cores summation
    cpu_cols = [c for c in df.columns if c.startswith('cpu_') and c.endswith('_percent')]
    if cpu_cols:
        df['cpu_cores_sum'] = df[cpu_cols].sum(axis=1)
        df['cpu_cores_sum_ma'] = df['cpu_cores_sum'].rolling(window=5, center=True, min_periods=1).mean()
    else:
        df['cpu_cores_sum'] = df.get('cpu_percent_total', 0)
        df['cpu_cores_sum_ma'] = df['cpu_cores_sum'].rolling(window=5, center=True, min_periods=1).mean()

    # Memory
    if 'memory_percent' not in df.columns:
        raise ValueError("Missing 'memory_percent' column in system metrics CSV")

    # GPU
    if 'gpu_0_util_percent' not in df.columns or 'gpu_0_mem_used_mb' not in df.columns:
        raise ValueError("Missing GPU columns in system metrics CSV")

    return df


def get_run_start_times(run_files):
    """Extract the start timestamp (user_bin==0) from each run stats CSV"""
    run_starts = []
    for rf in run_files:
        if not os.path.exists(rf):
            print(f"Warning: Run stats file '{rf}' not found, skipping.")
            continue
        try:
            r = pd.read_csv(rf, parse_dates=['bin_start_timestamp'])
            if 'user_bin' in r.columns:
                first = r.loc[r['user_bin'] == 0]
                ts = first['bin_start_timestamp'].iloc[0] if not first.empty else r['bin_start_timestamp'].iloc[0]
            else:
                ts = r['bin_start_timestamp'].iloc[0]
            run_starts.append(ts)
        except Exception as e:
            print(f"Warning: Could not parse '{rf}': {e}")
    return run_starts


def plot_cpu_data(df, run_starts):
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['cpu_cores_sum'], label='CPU Sum', linewidth=2, color='#e74c3c')
    plt.plot(df['datetime'], df['cpu_cores_sum_ma'], label='CPU 5-pt MA', linewidth=2, color='#3498db')
    for start in run_starts:
        plt.axvline(start, color='red', linestyle=':', linewidth=1.5)
    plt.title('CPU Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('CPU (%)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def plot_memory_data(df, run_starts):
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['memory_percent'], label='Memory Usage', linewidth=2, color='#2ecc71')
    for start in run_starts:
        plt.axvline(start, color='red', linestyle=':', linewidth=1.5)
    plt.title('Memory Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Memory (%)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def plot_gpu_utilization(df, run_starts):
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['gpu_0_util_percent'], label='GPU Utilization', linewidth=2, color='#9b59b6')
    for start in run_starts:
        plt.axvline(start, color='red', linestyle=':', linewidth=1.5)
    plt.title('GPU Utilization Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('GPU Util (%)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def plot_gpu_memory(df, run_starts):
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['gpu_0_mem_used_mb'], label='GPU Memory Used', linewidth=2, color='#e67e22')
    for start in run_starts:
        plt.axvline(start, color='red', linestyle=':', linewidth=1.5)
    plt.title('GPU Memory Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def create_summary_plot(df, run_starts):
    metrics = [
        ('cpu_cores_sum', 'CPU Sum', '%', '#e74c3c'),
        ('memory_percent', 'Memory', '%', '#2ecc71'),
        ('gpu_0_util_percent', 'GPU Util', '%', '#9b59b6'),
        ('gpu_0_mem_used_mb', 'GPU Mem', 'MB', '#e67e22')
    ]
    available = [(col,title,unit,color) for col,title,unit,color in metrics if col in df.columns]
    n = len(available)
    if n == 0:
        raise ValueError("No metrics available for summary plot")
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4*rows))
    axes = np.array(axes).reshape(-1)
    for ax, (col,title,unit,color) in zip(axes, available):
        ax.plot(df['datetime'], df[col], linewidth=2, color=color)
        for start in run_starts:
            idx = df['datetime'].searchsorted(start)
            ax.axvline(df['datetime'].iloc[idx] if idx < len(df) else start,
                       color='red', linestyle=':', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.get_xticklabels(), rotation=45)
    for ax in axes[len(available):]: ax.set_visible(False)
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) < 3:
        print("Usage: python resource_plot.py <system_metrics.csv> <run1.csv> [run2.csv ...]")
        sys.exit(1)
    sys_file = sys.argv[1]
    run_files = sys.argv[2:]
    if not os.path.exists(sys_file):
        print(f"Error: System metrics file '{sys_file}' not found!")
        sys.exit(1)

    # load data
    df = load_and_process_data(sys_file)
    run_starts = get_run_start_times(run_files)
    print(f"Found {len(run_starts)} run start times: {run_starts}")

    # generate plots
    figs = []
    figs.append(plot_cpu_data(df, run_starts))
    figs.append(plot_memory_data(df, run_starts))
    figs.append(plot_gpu_utilization(df, run_starts))
    figs.append(plot_gpu_memory(df, run_starts))
    figs.append(create_summary_plot(df, run_starts))

    # Save all plots by default
    names = ['cpu', 'memory', 'gpu_util', 'gpu_mem', 'summary']
    for name, fig in zip(names, figs):
        filename = f"{name}_plot.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    # Display plots
    plt.show()

if __name__ == "__main__":
    main()
