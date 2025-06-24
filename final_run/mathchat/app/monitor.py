import csv
import time
import subprocess
import threading
import psutil
import os

from codecarbon import EmissionsTracker

# --- Configuration ---
OUTPUT_FILE     = '/app/emissions_logs/system_metrics.csv'
SAMPLE_INTERVAL = 30  # seconds

def get_gpu_metrics():
    """
    Executes nvidia-smi to get GPU utilization and memory usage.
    Returns (util_list, mem_list) or (None, None) if unavailable.
    """
    try:
        smi_command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(smi_command, capture_output=True, text=True, check=True)
        gpu_util = []
        gpu_mem  = []
        for line in result.stdout.strip().split('\n'):
            util, mem = line.split(',')
            gpu_util.append(int(util.strip()))
            gpu_mem.append(int(mem.strip()))
        return gpu_util, gpu_mem

    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

def monitor_system(stop_event):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Probe how many GPUs / CPUs we have
    gpu_util, _ = get_gpu_metrics()
    num_gpus = len(gpu_util) if gpu_util is not None else 0
    num_cpus = psutil.cpu_count(logical=True)

    # Build CSV header
    header = [
        'timestamp',
        'cpu_percent_total',
        'memory_percent',
        'memory_used_mb',      # Added
        'memory_total_mb',     # Added
    ]
    # original GPU metrics (optional)
    for i in range(num_gpus):
        header += [f'gpu_{i}_util_percent', f'gpu_{i}_mem_used_mb']
    # CodeCarbon energy/emissions + metadata
    header += [
        'ram_power_w',
        'gpu_power_w',
        'emissions_kg',
        'country',
        'gpu_model',
        'cpu_model',
        'duration_sec',
        'cpu_power_w',
        'region',
    ]
    # per-core CPU%
    for i in range(num_cpus):
        header.append(f'cpu_{i}_percent')

    # Write header
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print(f"Monitoring started; logging every {SAMPLE_INTERVAL}s to {OUTPUT_FILE}")

    # Start a CodeCarbon tracker that we will stop/restart each loop
    cc_tracker = EmissionsTracker(
        project_name="final phase",
        output_dir=os.path.dirname(OUTPUT_FILE),
        measure_power_secs=1,
        log_level="ERROR",
    )
    cc_tracker.start()

    while not stop_event.is_set():
        try:
            # --- Sample system metrics ---
            timestamp = int(time.time() * 1000)
            cpu_total = psutil.cpu_percent(interval=1)
            
            # --- Enhanced Memory Capturing ---
            vmem = psutil.virtual_memory()
            mem_pct = vmem.percent
            mem_used_mb = vmem.used // (1024 * 1024)  # Convert bytes to MB
            mem_total_mb = vmem.total // (1024 * 1024) # Convert bytes to MB
            # --- End Enhanced Memory Capturing ---

            gpu_util, gpu_mem = get_gpu_metrics()
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

            # --- Stop & read CodeCarbon data for the last interval ---
            cc_tracker.stop()
            md = cc_tracker.final_emissions_data
            emissions_kg = cc_tracker.final_emissions

            # Extract CodeCarbon fields
            ram_power_w   = round(md.ram_power, 2)
            gpu_power_w   = round(md.gpu_power, 2)
            cpu_power_w   = round(md.cpu_power, 2)
            duration_sec  = round(md.duration, 3)
            country       = md.country_name
            region        = md.region
            gpu_model     = md.gpu_model
            cpu_model     = md.cpu_model

            # Restart tracker for next interval
            cc_tracker = EmissionsTracker(
                project_name="final phase",
                output_dir=os.path.dirname(OUTPUT_FILE),
                measure_power_secs=1,
                log_level="ERROR",
            )
            cc_tracker.start()

            # --- Build CSV row ---
            row = [timestamp, cpu_total, mem_pct, mem_used_mb, mem_total_mb] # Added memory metrics

            if gpu_util is not None:
                for i in range(num_gpus):
                    row += [gpu_util[i], gpu_mem[i]]

            # CodeCarbon columns
            row += [
                ram_power_w,
                gpu_power_w,
                round(emissions_kg, 6),
                country,
                gpu_model,
                cpu_model,
                duration_sec,
                cpu_power_w,
                region,
            ]

            # Per-core CPU%
            row += cpu_per_core

            # Append to file
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Wait remaining time (we already spent ~1s on cpu_percent)
            stop_event.wait(max(0, SAMPLE_INTERVAL - 1))

        except Exception as e:
            print(f"Monitoring error: {e}")
            stop_event.wait(SAMPLE_INTERVAL)

if __name__ == '__main__':
    stop_evt = threading.Event()
    t = threading.Thread(target=monitor_system, args=(stop_evt,))
    t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_evt.set()
        t.join()
        print("\nMonitoring stopped.")
