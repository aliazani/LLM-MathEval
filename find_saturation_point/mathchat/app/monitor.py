"""
System Resource Monitor Line Chart Plotting Script

This script creates clean, simple line charts for CPU, Memory, and GPU usage monitoring.
Each plot shows time series data with time on x-axis and the respective metric on y-axis.

📈 Charts Created:
1. CPU Usage: Time vs CPU % (sum of all cores + moving average)
2. Memory Usage: Time vs Memory %  
3. GPU Utilization: Time vs GPU %
4. GPU Memory: Time vs GPU Memory (MB)

🚀 Usage:
    python resource_plot.py <csv_file_path>
    
📋 Example:
    python resource_plot.py system_metrics.csv

🎯 Output:
- Clean line charts with time-series visualization
- X-axis: Time (HH:MM:SS format)
- Y-axis: Respective metric values
- Optional high-resolution PNG export
"""

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
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    required_columns = ['timestamp', 'memory_percent', 'gpu_0_util_percent', 'gpu_0_mem_used_mb']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['time_str'] = df['datetime'].dt.strftime('%H:%M:%S')
    else:
        print("Warning: No timestamp column found, using row index for time")
        df['time_str'] = [f"Point {i}" for i in range(len(df))]
    
    # Calculate sum of all CPU cores (cpu_0_percent to cpu_63_percent)
    cpu_columns = [col for col in df.columns if col.startswith('cpu_') and col.endswith('_percent') and col != 'cpu_percent_total']
    
    if cpu_columns:
        print(f"Found {len(cpu_columns)} CPU core columns")
        df['cpu_cores_sum'] = df[cpu_columns].sum(axis=1)
        df['cpu_cores_count'] = len(cpu_columns)
        df['cpu_max_possible'] = len(cpu_columns) * 100  # Each core can go up to 100%
        
        # Calculate moving average for CPU (5-point window)
        df['cpu_cores_sum_ma'] = df['cpu_cores_sum'].rolling(window=5, center=True, min_periods=1).mean()
    else:
        print("Warning: No individual CPU core columns found!")
        # Use cpu_percent_total if available, otherwise create dummy data
        if 'cpu_percent_total' in df.columns:
            df['cpu_cores_sum'] = df['cpu_percent_total']
            df['cpu_cores_count'] = 1
            df['cpu_max_possible'] = 100
            df['cpu_cores_sum_ma'] = df['cpu_cores_sum'].rolling(window=5, center=True, min_periods=1).mean()
        else:
            print("No CPU data available!")
            df['cpu_cores_sum'] = 0
            df['cpu_cores_count'] = 0
            df['cpu_max_possible'] = 100
            df['cpu_cores_sum_ma'] = 0
    
    # Add memory capacity info (assuming 100% is the limit)
    if 'memory_percent' in df.columns:
        df['memory_max_percent'] = 100
    
    # Add GPU capacity info (assuming 100% utilization is the limit)
    if 'gpu_0_util_percent' in df.columns:
        df['gpu_max_util_percent'] = 100
    
    return df

def plot_cpu_data(df):
    """Create CPU usage line chart"""
    
    plt.figure(figsize=(12, 6))
    
    # Plot CPU sum with moving average
    plt.plot(df['datetime'], df['cpu_cores_sum'], 
             label=f'CPU Sum (All {df["cpu_cores_count"].iloc[0]} Cores)', 
             linewidth=2, 
             color='#e74c3c')
    
    plt.plot(df['datetime'], df['cpu_cores_sum_ma'], 
             label='Moving Average (5-point)', 
             linewidth=2,
             color='#3498db')
    
    plt.title('CPU Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis to show time nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    return plt.gcf()

def plot_memory_data(df):
    """Create Memory usage line chart"""
    
    if 'memory_percent' not in df.columns:
        print("Warning: No memory_percent column found, skipping memory plot")
        return None
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['datetime'], df['memory_percent'], 
             linewidth=2,
             color='#2ecc71',
             label='Memory Usage')
    
    plt.title('Memory Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Memory Usage (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis to show time nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    return plt.gcf()

def plot_gpu_utilization(df):
    """Create GPU utilization line chart"""
    
    if 'gpu_0_util_percent' not in df.columns:
        print("Warning: No gpu_0_util_percent column found, skipping GPU utilization plot")
        return None
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['datetime'], df['gpu_0_util_percent'], 
             linewidth=2,
             color='#9b59b6',
             label='GPU Utilization')
    
    plt.title('GPU Utilization Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('GPU Utilization (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis to show time nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    return plt.gcf()

def plot_gpu_memory(df):
    """Create GPU memory usage line chart"""
    
    if 'gpu_0_mem_used_mb' not in df.columns:
        print("Warning: No gpu_0_mem_used_mb column found, skipping GPU memory plot")
        return None
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['datetime'], df['gpu_0_mem_used_mb'], 
             linewidth=2,
             color='#e67e22',
             label='GPU Memory Used')
    
    plt.title('GPU Memory Usage Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('GPU Memory (MB)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format x-axis to show time nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    return plt.gcf()

def create_summary_plot(df):
    """Create a summary plot with all metrics"""
    
    # Check which metrics are available
    has_cpu = 'cpu_cores_sum' in df.columns
    has_memory = 'memory_percent' in df.columns
    has_gpu_util = 'gpu_0_util_percent' in df.columns
    has_gpu_mem = 'gpu_0_mem_used_mb' in df.columns
    
    available_metrics = sum([has_cpu, has_memory, has_gpu_util, has_gpu_mem])
    
    if available_metrics == 0:
        print("Warning: No metrics available for summary plot")
        return None
    
    # Determine subplot layout
    if available_metrics == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        axes = [ax]
    elif available_metrics == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    elif available_metrics == 3:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        axes[-1].set_visible(False)  # Hide the last unused subplot
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
    
    plot_idx = 0
    
    # CPU plot
    if has_cpu:
        ax = axes[plot_idx]
        ax.plot(df.index, df['cpu_cores_sum'], label='CPU Sum', linewidth=2, color='#e74c3c')
        if 'cpu_cores_sum_ma' in df.columns:
            ax.plot(df.index, df['cpu_cores_sum_ma'], label='Moving Avg', linewidth=3, color='#3498db')
        ax.set_title('CPU Usage', fontweight='bold')
        ax.set_ylabel('CPU Usage (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Memory plot
    if has_memory:
        ax = axes[plot_idx]
        ax.plot(df.index, df['memory_percent'], linewidth=3, color='#2ecc71')
        ax.fill_between(df.index, df['memory_percent'], alpha=0.3, color='#2ecc71')
        ax.set_title('Memory Usage', fontweight='bold')
        ax.set_ylabel('Memory (%)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # GPU Utilization
    if has_gpu_util:
        ax = axes[plot_idx]
        ax.plot(df.index, df['gpu_0_util_percent'], linewidth=3, color='#9b59b6')
        ax.fill_between(df.index, df['gpu_0_util_percent'], alpha=0.3, color='#9b59b6')
        ax.set_title('GPU Utilization', fontweight='bold')
        ax.set_ylabel('GPU Util (%)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # GPU Memory
    if has_gpu_mem:
        ax = axes[plot_idx]
        ax.plot(df.index, df['gpu_0_mem_used_mb'], linewidth=3, color='#e67e22')
        ax.fill_between(df.index, df['gpu_0_mem_used_mb'], alpha=0.3, color='#e67e22')
        ax.set_title('GPU Memory Usage', fontweight='bold')
        ax.set_ylabel('GPU Memory (MB)')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Set x-axis ticks for all subplots
    tick_positions = np.linspace(0, len(df)-1, min(8, len(df)), dtype=int)
    for i in range(plot_idx):
        ax = axes[i]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([df['time_str'].iloc[j] for j in tick_positions], rotation=45)
        # Add xlabel to bottom plots
        if (available_metrics <= 2) or (i >= available_metrics - 2):
            ax.set_xlabel('Time Point')
    
    plt.suptitle('System Performance Overview', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

def main():
    """Main function to create all plots"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python resource_plot.py <csv_file_path>")
        print("Example: python resource_plot.py system_metrics.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        print(f"Please check the file path and try again.")
        sys.exit(1)
    
    print(f"Loading and processing data from: {file_path}")
    try:
        df = load_and_process_data(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please make sure the file is a valid CSV with the expected format.")
        sys.exit(1)
    
    print(f"Data loaded successfully!")
    if 'datetime' in df.columns:
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Number of data points: {len(df)}")
    print(f"CPU cores sum range: {df['cpu_cores_sum'].min():.1f}% - {df['cpu_cores_sum'].max():.1f}%")
    if 'memory_percent' in df.columns:
        print(f"Memory range: {df['memory_percent'].min():.1f}% - {df['memory_percent'].max():.1f}%")
    if 'gpu_0_util_percent' in df.columns:
        print(f"GPU utilization range: {df['gpu_0_util_percent'].min():.1f}% - {df['gpu_0_util_percent'].max():.1f}%")
    
    # Create separate line chart plots
    plots_created = []
    
    print("\n" + "="*50)
    print("📈 CREATING SIMPLE LINE CHARTS")
    print("="*50)
    
    print("\n1️⃣ Creating CPU line chart...")
    cpu_fig = plot_cpu_data(df)
    if cpu_fig is not None:
        plots_created.append(('CPU', cpu_fig))
        print("✅ CPU line chart created!")
    
    print("\n2️⃣ Creating Memory line chart...")
    memory_fig = plot_memory_data(df)
    if memory_fig is not None:
        plots_created.append(('Memory', memory_fig))
        print("✅ Memory line chart created!")
    
    print("\n3️⃣ Creating GPU Utilization line chart...")
    gpu_util_fig = plot_gpu_utilization(df)
    if gpu_util_fig is not None:
        plots_created.append(('GPU_Utilization', gpu_util_fig))
        print("✅ GPU Utilization line chart created!")
    
    print("\n4️⃣ Creating GPU Memory line chart...")
    gpu_mem_fig = plot_gpu_memory(df)
    if gpu_mem_fig is not None:
        plots_created.append(('GPU_Memory', gpu_mem_fig))
        print("✅ GPU Memory line chart created!")
    
    if not plots_created:
        print("❌ No plots could be created due to missing data columns.")
        return
    
    print(f"\n🎉 Successfully created {len(plots_created)} line charts!")
    print("📊 Each plot shows time series data with:")
    print("   • X-axis: Time")
    print("   • Y-axis: Respective metric values")
    print("   • Clean line visualization")
    
    # Show all plots
    plt.show()
    
    # Optionally save plots
    save_plots = input(f"\n💾 Do you want to save the {len(plots_created)} plots as PNG files? (y/n): ").lower().strip()
    if save_plots == 'y':
        print("\n📁 Saving plots...")
        for plot_name, fig in plots_created:
            filename = f'{plot_name.lower()}_line_chart.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
        print("🎉 All line charts saved successfully!")

if __name__ == "__main__":
    main()