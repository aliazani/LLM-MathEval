#!/usr/bin/env python3
"""
Download CSV data from Phoenix server
"""

import os
import sys

def download_csv_from_phoenix():
    """Download spans dataframe from Phoenix and save as CSV"""
    try:
        import phoenix as px
        
        # Phoenix server endpoint (configurable via env var)
        phoenix_endpoint = os.getenv("PHOENIX_URL", "http://localhost:6006")
        project_name = os.getenv("PHOENIX_PROJECT", "vllm-custom-server-final")
        output_file = os.getenv("CSV_OUTPUT", "/app/vllm_spans.csv")
        
        print(f"Connecting to Phoenix at: {phoenix_endpoint}")
        print(f"Project: {project_name}")
        
        client = px.Client(endpoint=phoenix_endpoint)
        
        print("Downloading spans dataframe...")
        df = client.get_spans_dataframe(
            project_name=project_name,
            limit=1_000_000,      # bigger than your 965 608 rows
            timeout=None          # wait as long as it takes
        )
        
        print(f"Downloaded {len(df):,} rows")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✓ Wrote {len(df):,} rows to {output_file}")
        
        return output_file
        
    except ImportError:
        print("ERROR: arize-phoenix package not found")
        print("Install with: pip install arize-phoenix")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR downloading data from Phoenix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_csv_from_phoenix()
