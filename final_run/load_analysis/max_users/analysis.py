import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def parse_jtl(jtl_file):
    # Read CSV, handle bad lines compatibly across pandas versions
    read_csv_kwargs = {
        'delimiter': ',',
        'header': 0,
        'engine': 'python'
    }
    try:
        read_csv_kwargs['on_bad_lines'] = 'skip'
        df = pd.read_csv(jtl_file, **read_csv_kwargs)
    except TypeError:
        read_csv_kwargs.pop('on_bad_lines', None)
        df = pd.read_csv(jtl_file, error_bad_lines=False, **read_csv_kwargs)

    # Rename columns
    df.rename(columns={
        'timeStamp': 'timestamp', 'elapsed': 'elapsed', 'label': 'label',
        'responseCode': 'responsecode', 'responseMessage': 'responsemessage',
        'threadName': 'threadname', 'dataType': 'datatype', 'success': 'success',
        'failureMessage': 'failuremessage', 'bytes': 'bytes', 'sentBytes': 'sentbytes',
        'grpThreads': 'grpthreads', 'allThreads': 'allthreads', 'URL': 'url',
        'Latency': 'latency', 'Hostname': 'hostname', 'IdleTime': 'idletime',
        'Connect': 'connect'
    }, inplace=True)

    # Filter HTTP Request rows
    df = df[df['label'] == 'HTTP Request']

    # Clean numeric
    for col in ['timestamp', 'elapsed', 'latency', 'allthreads']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['timestamp', 'elapsed', 'latency', 'allthreads'], inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['success'] = df['success'].astype(str).str.lower() == 'true'
    return df


def compute_stats(df):
    stats = df.groupby('allthreads').agg(
        total_requests=('success', 'count'),
        failed_requests=('success', lambda x: (~x).sum()),
        error_rate=('success', lambda x: 100 * (~x).sum() / len(x)),
        avg_elapsed=('elapsed', 'mean')
    ).reset_index()
    return stats.sort_values('allthreads')


def determine_nmax_binned(stats, df, bin_width=10, error_rate_threshold=5.0):
    stats['user_bin'] = (stats['allthreads'] // bin_width) * bin_width

    # Create user_bin column in original dataframe to get timestamps
    df['user_bin'] = (df['allthreads'] // bin_width) * bin_width

    # Get the start timestamp for each user bin
    bin_start_times = df.groupby('user_bin')['timestamp'].min().reset_index()
    bin_start_times.rename(columns={'timestamp': 'bin_start_timestamp'}, inplace=True)

    # Aggregate stats by user bin
    binned = stats.groupby('user_bin').agg(
        avg_error_rate=('error_rate', 'mean'),
        avg_elapsed=('avg_elapsed', 'mean'),
        request_count=('error_rate', 'count')
    ).reset_index()

    # Merge with start timestamps
    binned = binned.merge(bin_start_times, on='user_bin', how='left')

    # Reorder columns to put timestamp near the beginning
    column_order = ['user_bin', 'bin_start_timestamp', 'avg_error_rate', 'avg_elapsed', 'request_count']
    binned = binned[column_order]

    breaking = binned[binned['avg_error_rate'] > error_rate_threshold]
    nmax = int(breaking.iloc[0]['user_bin']) if not breaking.empty else int(binned['user_bin'].max())
    return binned, nmax


def plot_and_save(binned, nmax, output_prefix='results'):
    """
    Create visually enhanced plots and save CSV/figures.
    """
    sns.set_theme(style='whitegrid')

    # Save aggregated stats
    stats_csv = f"{output_prefix}_stats.csv"
    binned.to_csv(stats_csv, index=False)
    print(f"Stats written to {stats_csv}")

    # --- Error Rate Plot ---
    plt.figure(figsize=(10, 5))
    x = binned['user_bin']
    y_err = binned['avg_error_rate']
    # Smooth line and fill
    sns.lineplot(x=x, y=y_err, color='tab:blue', linewidth=3, alpha=0.8, label='Error Rate (%)')
    plt.fill_between(x, y_err, alpha=0.2, color='tab:blue')
    plt.axvline(nmax, color='red', linestyle='--', linewidth=2, label=f'Nmax = {nmax}')
    plt.xlabel('User Bin')
    plt.ylabel('Avg Error Rate (%)')
    plt.title('Average Error Rate per User Bin')
    plt.legend()
    plt.tight_layout()
    err_file = f"{output_prefix}_error_rate.png"
    plt.savefig(err_file, dpi=150)
    print(f"Error rate plot saved to {err_file}")
    plt.close()

    # --- Elapsed Time Plot ---
    plt.figure(figsize=(10, 5))
    y_lat = binned['avg_elapsed'] / 60000  # to minutes
    sns.lineplot(x=x, y=y_lat, color='tab:orange', linewidth=3, alpha=0.8, label='Avg Elapsed Time (min)')
    plt.fill_between(x, y_lat, alpha=0.2, color='tab:orange')
    plt.axvline(nmax, color='red', linestyle='--', linewidth=2, label=f'Nmax = {nmax}')
    plt.xlabel('User Bin')
    plt.ylabel('Avg Elapsed Time (minutes)')
    plt.title('Average Elapsed Time per User Bin')
    plt.legend()
    plt.tight_layout()
    lat_file = f"{output_prefix}_elapsed_time.png"
    plt.savefig(lat_file, dpi=150)
    print(f"Elapsed time plot saved to {lat_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze JTL and save enhanced CSV/plots')
    parser.add_argument('jtl_file', help='Path to the JTL file')
    parser.add_argument('--prefix', default='results', help='Output file prefix')
    args = parser.parse_args()

    df = parse_jtl(args.jtl_file)
    stats = compute_stats(df)
    binned, nmax = determine_nmax_binned(stats, df)
    plot_and_save(binned, nmax, output_prefix=args.prefix)

if __name__ == '__main__':
    main()