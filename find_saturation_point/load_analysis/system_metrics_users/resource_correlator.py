"""
Resource Usage vs Load Test Correlation Script

This script correlates load test user bins with system resource usage to show
how CPU, Memory, and GPU metrics change at different user load levels.

📊 Features:
- Correlates user load bins with system resource metrics by timestamp
- Shows resource usage at different user load levels
- Plots every 100 users for better readability
- Creates separate plots for each resource type

🚀 Usage:
    python resource_correlator.py <user_bins_csv> <system_metrics_csv>
    
📋 Example:
    python resource_correlator.py user_bins.csv system_metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import argparse
import sys
import os
from datetime import datetime, timedelta


def load_user_bins(csv_file):
    """Load user bin data with timestamps"""
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_cols = ['user_bin', 'bin_start_timestamp', 'avg_error_rate', 'avg_elapsed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in user bins file: {missing_cols}")
            return None
            
        # Convert timestamp to datetime
        df['bin_start_timestamp'] = pd.to_datetime(df['bin_start_timestamp'])
        
        # Filter to every 100 users for better readability
        df_filtered = df[df['user_bin'] % 100 == 0].copy()
        
        print(f"Loaded {len(df)} user bin records")
        print(f"Filtered to {len(df_filtered)} records (every 100 users)")
        print(f"User load range: {df['user_bin'].min()} - {df['user_bin'].max()}")
        print(f"Time range: {df['bin_start_timestamp'].min()} to {df['bin_start_timestamp'].max()}")
        
        return df_filtered
        
    except Exception as e:
        print(f"Error loading user bins file: {e}")
        return None


def load_system_metrics(csv_file):
    """Load system metrics data"""
    try:
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            print("Warning: No timestamp column found in system metrics")
            return None
            
        print(f"Loaded {len(df)} system metric records")
        print(f"System metrics time range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading system metrics file: {e}")
        return None


def correlate_data(user_bins_df, system_df, time_tolerance_seconds=30):
    """
    Correlate user bins with system metrics based on timestamps
    """
    print(f"Correlating data with {time_tolerance_seconds}s tolerance...")
    
    correlated_data = []
    
    for _, user_row in user_bins_df.iterrows():
        user_time = user_row['bin_start_timestamp']
        user_bin = user_row['user_bin']
        
        # Find system metrics within time tolerance
        time_diff = abs(system_df['datetime'] - user_time)
        closest_idx = time_diff.idxmin()
        closest_time_diff = time_diff.loc[closest_idx].total_seconds()
        
        if closest_time_diff <= time_tolerance_seconds:
            sys_row = system_df.loc[closest_idx]
            
            # Calculate CPU sum if individual cores exist
            cpu_sum = 0
            cpu_cores = [col for col in system_df.columns if col.startswith('cpu_') and col.endswith('_percent') and col != 'cpu_percent_total']
            if cpu_cores:
                cpu_sum = sys_row[cpu_cores].sum()
            elif 'cpu_percent_total' in system_df.columns:
                cpu_sum = sys_row['cpu_percent_total']
            
            correlated_data.append({
                'user_bin': user_bin,
                'bin_start_timestamp': user_time,
                'system_timestamp': sys_row['datetime'],
                'time_diff_seconds': closest_time_diff,
                'avg_error_rate': user_row['avg_error_rate'],
                'avg_elapsed': user_row['avg_elapsed'],
                'cpu_usage': cpu_sum,
                'memory_percent': sys_row.get('memory_percent', np.nan),
                'gpu_util_percent': sys_row.get('gpu_0_util_percent', np.nan),
                'gpu_mem_used_mb': sys_row.get('gpu_0_mem_used_mb', np.nan),
                'cpu_cores_count': len(cpu_cores) if cpu_cores else 1
            })
        else:
            print(f"Warning: No system data within {time_tolerance_seconds}s for user bin {user_bin} at {user_time}")
    
    result_df = pd.DataFrame(correlated_data)
    print(f"Successfully correlated {len(result_df)} data points")
    
    return result_df


def create_correlation_plots(correlated_df, output_prefix='correlation'):
    """
    Create plots showing resource usage vs user load
    """
    print("Creating correlation plots...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. CPU Usage vs User Load
    plt.figure(figsize=(12, 6))
    plt.plot(correlated_df['user_bin'], correlated_df['cpu_usage'], 
             'o-', linewidth=2, markersize=6, color='#e74c3c', label='CPU Usage')
    plt.xlabel('User Load (Concurrent Users)', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title(f'CPU Usage vs User Load (All {correlated_df["cpu_cores_count"].iloc[0]} Cores)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations for key points
    for i in range(0, len(correlated_df), max(1, len(correlated_df)//5)):
        row = correlated_df.iloc[i]
        plt.annotate(f'{row["cpu_usage"]:.0f}%', 
                    xy=(row['user_bin'], row['cpu_usage']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    cpu_file = f"{output_prefix}_cpu_vs_load.png"
    plt.savefig(cpu_file, dpi=300, bbox_inches='tight')
    print(f"CPU correlation plot saved to {cpu_file}")
    plt.close()
    
    # 2. Memory Usage vs User Load
    if not correlated_df['memory_percent'].isna().all():
        plt.figure(figsize=(12, 6))
        plt.plot(correlated_df['user_bin'], correlated_df['memory_percent'], 
                 'o-', linewidth=2, markersize=6, color='#2ecc71', label='Memory Usage')
        plt.xlabel('User Load (Concurrent Users)', fontsize=12)
        plt.ylabel('Memory Usage (%)', fontsize=12)
        plt.title('Memory Usage vs User Load', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        for i in range(0, len(correlated_df), max(1, len(correlated_df)//5)):
            row = correlated_df.iloc[i]
            if not pd.isna(row['memory_percent']):
                plt.annotate(f'{row["memory_percent"]:.1f}%', 
                            xy=(row['user_bin'], row['memory_percent']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        mem_file = f"{output_prefix}_memory_vs_load.png"
        plt.savefig(mem_file, dpi=300, bbox_inches='tight')
        print(f"Memory correlation plot saved to {mem_file}")
        plt.close()
    
    # 3. GPU Utilization vs User Load
    if not correlated_df['gpu_util_percent'].isna().all():
        plt.figure(figsize=(12, 6))
        plt.plot(correlated_df['user_bin'], correlated_df['gpu_util_percent'], 
                 'o-', linewidth=2, markersize=6, color='#9b59b6', label='GPU Utilization')
        plt.xlabel('User Load (Concurrent Users)', fontsize=12)
        plt.ylabel('GPU Utilization (%)', fontsize=12)
        plt.title('GPU Utilization vs User Load', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        for i in range(0, len(correlated_df), max(1, len(correlated_df)//5)):
            row = correlated_df.iloc[i]
            if not pd.isna(row['gpu_util_percent']):
                plt.annotate(f'{row["gpu_util_percent"]:.1f}%', 
                            xy=(row['user_bin'], row['gpu_util_percent']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        gpu_util_file = f"{output_prefix}_gpu_util_vs_load.png"
        plt.savefig(gpu_util_file, dpi=300, bbox_inches='tight')
        print(f"GPU utilization correlation plot saved to {gpu_util_file}")
        plt.close()
    
    # 4. GPU Memory vs User Load
    if not correlated_df['gpu_mem_used_mb'].isna().all():
        plt.figure(figsize=(12, 6))
        plt.plot(correlated_df['user_bin'], correlated_df['gpu_mem_used_mb'], 
                 'o-', linewidth=2, markersize=6, color='#e67e22', label='GPU Memory')
        plt.xlabel('User Load (Concurrent Users)', fontsize=12)
        plt.ylabel('GPU Memory (MB)', fontsize=12)
        plt.title('GPU Memory Usage vs User Load', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        for i in range(0, len(correlated_df), max(1, len(correlated_df)//5)):
            row = correlated_df.iloc[i]
            if not pd.isna(row['gpu_mem_used_mb']):
                plt.annotate(f'{row["gpu_mem_used_mb"]:.0f}MB', 
                            xy=(row['user_bin'], row['gpu_mem_used_mb']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        gpu_mem_file = f"{output_prefix}_gpu_memory_vs_load.png"
        plt.savefig(gpu_mem_file, dpi=300, bbox_inches='tight')
        print(f"GPU memory correlation plot saved to {gpu_mem_file}")
        plt.close()
    
    # 5. Combined Performance Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Error Rate
    ax1.plot(correlated_df['user_bin'], correlated_df['avg_error_rate'], 
             'o-', linewidth=2, markersize=4, color='#e74c3c')
    ax1.set_xlabel('User Load')
    ax1.set_ylabel('Error Rate (%)')
    ax1.set_title('Error Rate vs User Load')
    ax1.grid(True, alpha=0.3)
    
    # Response Time
    ax2.plot(correlated_df['user_bin'], correlated_df['avg_elapsed'], 
             'o-', linewidth=2, markersize=4, color='#f39c12')
    ax2.set_xlabel('User Load')
    ax2.set_ylabel('Avg Response Time (ms)')
    ax2.set_title('Response Time vs User Load')
    ax2.grid(True, alpha=0.3)
    
    # CPU Usage
    ax3.plot(correlated_df['user_bin'], correlated_df['cpu_usage'], 
             'o-', linewidth=2, markersize=4, color='#3498db')
    ax3.set_xlabel('User Load')
    ax3.set_ylabel('CPU Usage (%)')
    ax3.set_title('CPU Usage vs User Load')
    ax3.grid(True, alpha=0.3)
    
    # Memory Usage
    if not correlated_df['memory_percent'].isna().all():
        ax4.plot(correlated_df['user_bin'], correlated_df['memory_percent'], 
                 'o-', linewidth=2, markersize=4, color='#2ecc71')
        ax4.set_xlabel('User Load')
        ax4.set_ylabel('Memory Usage (%)')
        ax4.set_title('Memory Usage vs User Load')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Memory Data Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Memory Usage vs User Load')
    
    plt.suptitle('Performance and Resource Correlation Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    combined_file = f"{output_prefix}_combined_overview.png"
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"Combined overview plot saved to {combined_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Correlate load test user bins with system resource usage')
    parser.add_argument('user_bins_csv', help='Path to user bins CSV file')
    parser.add_argument('system_metrics_csv', help='Path to system metrics CSV file')
    parser.add_argument('--output-prefix', default='correlation', help='Output file prefix')
    parser.add_argument('--time-tolerance', type=int, default=30, help='Time tolerance in seconds for correlation')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.user_bins_csv):
        print(f"Error: User bins file '{args.user_bins_csv}' not found!")
        sys.exit(1)
        
    if not os.path.exists(args.system_metrics_csv):
        print(f"Error: System metrics file '{args.system_metrics_csv}' not found!")
        sys.exit(1)
    
    print("="*60)
    print("🔗 LOAD TEST vs SYSTEM RESOURCE CORRELATION")
    print("="*60)
    
    # Load data
    print("\n1️⃣ Loading user bin data...")
    user_bins_df = load_user_bins(args.user_bins_csv)
    if user_bins_df is None:
        sys.exit(1)
    
    print("\n2️⃣ Loading system metrics data...")
    system_df = load_system_metrics(args.system_metrics_csv)
    if system_df is None:
        sys.exit(1)
    
    # Correlate data
    print("\n3️⃣ Correlating data by timestamps...")
    correlated_df = correlate_data(user_bins_df, system_df, args.time_tolerance)
    if correlated_df.empty:
        print("Error: No data could be correlated!")
        sys.exit(1)
    
    # Save correlated data
    output_csv = f"{args.output_prefix}_correlated_data.csv"
    correlated_df.to_csv(output_csv, index=False)
    print(f"\n📁 Correlated data saved to: {output_csv}")
    
    # Create plots
    print("\n4️⃣ Creating correlation plots...")
    create_correlation_plots(correlated_df, args.output_prefix)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CORRELATION SUMMARY")
    print("="*60)
    print(f"Total correlated data points: {len(correlated_df)}")
    print(f"User load range: {correlated_df['user_bin'].min()} - {correlated_df['user_bin'].max()}")
    print(f"CPU usage range: {correlated_df['cpu_usage'].min():.1f}% - {correlated_df['cpu_usage'].max():.1f}%")
    if not correlated_df['memory_percent'].isna().all():
        print(f"Memory usage range: {correlated_df['memory_percent'].min():.1f}% - {correlated_df['memory_percent'].max():.1f}%")
    if not correlated_df['gpu_util_percent'].isna().all():
        print(f"GPU utilization range: {correlated_df['gpu_util_percent'].min():.1f}% - {correlated_df['gpu_util_percent'].max():.1f}%")
    
    print("\n🎉 Correlation analysis complete!")


if __name__ == "__main__":
    main()