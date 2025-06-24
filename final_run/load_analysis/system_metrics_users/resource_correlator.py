#!/usr/bin/env python3
"""
Resource Usage vs Combined Load Test Correlation Script

Usage:
    python resource_correlator.py <system_metrics_csv> <run1_bins.csv> [<run2_bins.csv> ...]

This script:
 1. Loads system metrics and any number of load-test bin CSVs (runs).
 2. Correlates each run’s user bins with nearest system metrics within tolerance.
 3. Produces combined plots per metric, stitching runs end-to-end,
    drawing vertical red-dotted lines at each run start,
    and labeling separators 0…N.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from datetime import timedelta


def load_user_bins(path):
    df = pd.read_csv(path)
    df['bin_start_timestamp'] = pd.to_datetime(df['bin_start_timestamp'])
    df = df[df['user_bin'] % 100 == 0]
    return df


def load_system(path):
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        raise ValueError("No 'timestamp' column in system metrics file")
    return df


def load_period_start(path):
    df = pd.read_csv(path, parse_dates=['bin_start_timestamp'])
    if df.empty:
        raise ValueError(f"Period file {path} is empty")
    return df['bin_start_timestamp'].iloc[0]


def correlate_run(user_df, sys_df, start_time, end_time, tol):
    user_df = user_df[(user_df['bin_start_timestamp'] >= start_time) &
                      (user_df['bin_start_timestamp'] <= end_time)]
    sys_df  = sys_df [(sys_df['datetime'] >= start_time) &
                     (sys_df['datetime'] <= end_time)]

    if user_df.empty or sys_df.empty:
        return pd.DataFrame()

    records = []
    cpu_cols = [c for c in sys_df.columns if c.startswith('cpu_') and c.endswith('_percent')]
    for _, u in user_df.iterrows():
        t0 = u['bin_start_timestamp']
        diffs = (sys_df['datetime'] - t0).abs()
        idx  = diffs.idxmin()
        if diffs.loc[idx].total_seconds() <= tol:
            s = sys_df.loc[idx]
            cpu = s[cpu_cols].sum() if cpu_cols else s.get('cpu_percent_total', np.nan)
            records.append({
                'user_bin':      u['user_bin'],
                'avg_error_rate': u['avg_error_rate'],
                'avg_elapsed':   u['avg_elapsed'],
                'cpu_usage':     cpu,
                'memory_percent': s.get('memory_percent', np.nan),
                'gpu_util':      s.get('gpu_0_util_percent', np.nan),
                'gpu_mem':       s.get('gpu_0_mem_used_mb', np.nan),
            })
    return pd.DataFrame(records)


def plot_combined(all_dfcs, offsets, prefix):
    metrics = [
        ('cpu_usage',     'CPU Usage (%)',       'CPU vs Load'),
        ('memory_percent','Memory (%)',          'Memory vs Load'),
        ('gpu_util',      'GPU Util (%)',        'GPU Util vs Load'),
        ('gpu_mem',       'GPU Mem (MB)',        'GPU Mem vs Load'),
        ('avg_error_rate','Error Rate',          'Error Rate vs Load'),
        ('avg_elapsed',   'Avg Elapsed (ms)',    'Response Time vs Load'),
    ]

    for metric, ylabel, title in metrics:
        plt.figure(figsize=(10,6))
        for idx, (dfc, off) in enumerate(zip(all_dfcs, offsets)):
            x = dfc['user_bin'] + off
            y = dfc[metric]
            plt.plot(x, y, 'o-', label=f'Run {idx}')
        ymin, ymax = plt.ylim()
        for idx, off in enumerate(offsets):
            plt.axvline(off, color='red', linestyle=':', linewidth=1.5)
            plt.text(off, ymin, str(idx), color='red', va='bottom', ha='center')
        plt.xlabel('Global "Users" (concatenated)')
        plt.ylabel(ylabel)
        plt.title(f"{title} (combined runs)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_{metric}_combined.png")
        print(f"Saved combined plot: {prefix}_{metric}_combined.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Correlate multiple runs and plot combined metrics'
    )
    parser.add_argument('system_metrics_csv', help='CSV with timestamp (ms) and system metrics')
    parser.add_argument('run_bins_csvs', nargs='+', help='One or more load-test bin CSVs')
    parser.add_argument('--tol', type=int, default=30, help='Time tolerance in seconds')
    parser.add_argument('--output-prefix', default='correlation', help='Output file prefix')
    args = parser.parse_args()

    if not os.path.exists(args.system_metrics_csv):
        sys.exit(f"Error: '{args.system_metrics_csv}' not found")
    sys_df = load_system(args.system_metrics_csv)

    starts = [load_period_start(f) for f in args.run_bins_csvs]
    ends   = [
        (starts[i+1] - timedelta(minutes=5)) if i+1 < len(starts) else pd.Timestamp.max
        for i in range(len(starts))
    ]

    all_dfcs = []
    offsets = []
    cum_offset = 0

    for i, (bins_csv, start, end) in enumerate(zip(args.run_bins_csvs, starts, ends)):
        user_df = load_user_bins(bins_csv)
        dfc = correlate_run(user_df, sys_df, start, end, args.tol)
        if dfc.empty:
            continue
        # accumulate
        all_dfcs.append(dfc)
        offsets.append(cum_offset)
        cum_offset += int(dfc['user_bin'].max())

    if all_dfcs:
        plot_combined(all_dfcs, offsets, args.output_prefix)
    else:
        print("No correlated data to plot.")

if __name__ == '__main__':
    main()
