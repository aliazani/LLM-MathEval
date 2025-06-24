#!/usr/bin/env python3
"""
Emissions vs Load Test Correlation Script

Usage:
    python emissions_correlator.py <emissions_csv> <run1_bins.csv> [<run2_bins.csv> ...]

This script:
 1. Loads a CSV of emissions/power/energy metrics (with a 'timestamp' column).
 2. Loads one or more load-test bin CSVs (each with 'bin_start_timestamp' and 'user_bin').
 3. For each run, correlates each user bin timestamp to the nearest emissions record (within tolerance).
 4. Combines all runs end-to-end and generates time-series plots of:
    emissions_rate, emissions,
    cpu_power, gpu_power, ram_power,
    cpu_energy, gpu_energy, ram_energy,
    energy_consumed
 5. Draws vertical red-dotted lines separating runs (labeled 0…N).

Example:
    python emissions_correlator.py emissions.csv run1.csv run2.csv run3.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from datetime import timedelta


def load_emissions(path):
    df = pd.read_csv(path)
    if 'timestamp' not in df:
        raise ValueError("Emissions CSV must have a 'timestamp' column")
    df['datetime'] = pd.to_datetime(df['timestamp'])
    return df


def load_user_bins(path):
    df = pd.read_csv(path)
    if 'bin_start_timestamp' not in df:
        raise ValueError(f"Run CSV {path} lacks 'bin_start_timestamp'")
    df['bin_start_timestamp'] = pd.to_datetime(df['bin_start_timestamp'])
    df = df[df['user_bin'] % 100 == 0]
    return df


def load_period_start(path):
    df = pd.read_csv(path, parse_dates=['bin_start_timestamp'])
    if df.empty:
        raise ValueError(f"Run CSV {path} is empty")
    return df['bin_start_timestamp'].iloc[0]


def correlate_run(user_df, emis_df, start_time, end_time, tol):
    ud = user_df[(user_df['bin_start_timestamp'] >= start_time) &
                 (user_df['bin_start_timestamp'] <= end_time)]
    ed = emis_df[(emis_df['datetime'] >= start_time) &
                 (emis_df['datetime'] <= end_time)]
    if ud.empty or ed.empty:
        return pd.DataFrame()

    recs = []
    metrics = ['emissions_rate','emissions','cpu_power','gpu_power','ram_power',
               'cpu_energy','gpu_energy','ram_energy','energy_consumed']
    for _, u in ud.iterrows():
        t0 = u['bin_start_timestamp']
        diffs = (ed['datetime'] - t0).abs()
        idx = diffs.idxmin()
        if diffs.loc[idx].total_seconds() <= tol:
            e = ed.loc[idx]
            row = {'user_bin': u['user_bin']}
            for m in metrics:
                row[m] = e.get(m, np.nan)
            recs.append(row)
    return pd.DataFrame(recs)


def plot_combined(all_dfcs, offsets, prefix):
    metrics = ['emissions_rate','emissions','cpu_power','gpu_power','ram_power',
               'cpu_energy','gpu_energy','ram_energy','energy_consumed']
    for metric in metrics:
        plt.figure(figsize=(10,6))
        for idx, (dfc, off) in enumerate(zip(all_dfcs, offsets)):
            x = dfc['user_bin'] + off
            y = dfc[metric]
            plt.plot(x, y, 'o-', label=f'Run {idx}')
        ymin, ymax = plt.ylim()
        for idx, off in enumerate(offsets):
            plt.axvline(off, color='red', linestyle=':', linewidth=1.5)
            plt.text(off, ymin, str(idx), color='red', va='bottom', ha='center')
        plt.xlabel('Global Users (concatenated)')
        plt.ylabel(metric.replace('_',' ').title())
        plt.title(f"{metric.replace('_',' ').title()} vs Load (combined)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = f"{prefix}_{metric}_combined.png"
        plt.savefig(out)
        print(f"Saved combined plot: {out}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Correlate emissions metrics with load-test bins and plot combined runs'
    )
    parser.add_argument('emissions_csv', help='CSV with timestamped emissions/power/energy data')
    parser.add_argument('run_bins_csvs', nargs='+', help='One or more run bin CSVs')
    parser.add_argument('--tol', type=int, default=30, help='Time tolerance in seconds')
    parser.add_argument('--output-prefix', default='emissions',
                        help='Prefix for output files')
    args = parser.parse_args()

    if not os.path.exists(args.emissions_csv):
        sys.exit(f"Error: Emissions file '{args.emissions_csv}' not found")
    emis_df = load_emissions(args.emissions_csv)

    # determine windows
    starts = [load_period_start(f) for f in args.run_bins_csvs]
    ends = [
        (starts[i+1] - timedelta(minutes=5)) if i+1 < len(starts) else pd.Timestamp.max
        for i in range(len(starts))
    ]

    all_dfcs, offsets = [], []
    cum_off = 0
    for i, (run_csv, start, end) in enumerate(zip(args.run_bins_csvs, starts, ends)):
        if not os.path.exists(run_csv):
            print(f"Warning: Run file '{run_csv}' not found; skipping")
            continue
        user_df = load_user_bins(run_csv)
        dfc = correlate_run(user_df, emis_df, start, end, args.tol)
        if dfc.empty:
            print(f"No data for run {i}; skipping")
            continue
        all_dfcs.append(dfc)
        offsets.append(cum_off)
        cum_off += int(dfc['user_bin'].max())

    if all_dfcs:
        plot_combined(all_dfcs, offsets, args.output_prefix)
    else:
        print("No correlated data to plot.")

if __name__ == '__main__':
    main()
