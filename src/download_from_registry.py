"""
File: src/download_from_registry.py
Description: 
    This script is for TEAMMATES to sync data.
    It reads the 'metadata_registry.json' (committed to git) and downloads
    any CSV files that are missing locally.
    
    Why? 
    Because CSV files are gitignored. Teammates pull the registry but need 
    to fetch the actual files to do their work.
    
    Usage: python src/download_from_registry.py
"""

import os
import json
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv

# --- Configuration ---
DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "metadata_registry.json")
DOWNLOAD_ROWS_LIMIT = 100 

# 1. Load environment variables
load_dotenv()
SOCRATA_TOKEN = os.getenv("SOCRATA_APP_TOKEN")

if not SOCRATA_TOKEN:
    print(" Error: SOCRATA_APP_TOKEN not found in .env file.")
    exit()

def sync_data_from_registry():
    print("-" * 50)
    print(" Starting Data Sync (Registry -> Local CSVs)")
    print("-" * 50)

    # Check if registry exists
    if not os.path.exists(METADATA_FILE):
        print(f" Error: {METADATA_FILE} not found.")
        print("   Please pull from git or run 'data_collector.py' first.")
        return

    # Load Registry
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    print(f" Registry contains {len(registry)} datasets.")
    
    client = Socrata("data.cityofnewyork.us", SOCRATA_TOKEN)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    downloaded_count = 0
    skipped_count = 0

    for i, item in enumerate(registry):
        dataset_id = item['id']
        # Use the path from registry, or fallback to default naming
        csv_filename = f"{dataset_id}.csv"
        csv_path = os.path.join(DATA_DIR, csv_filename)
        
        # 1. Check if file already exists
        if os.path.exists(csv_path):
            # Optional: Check if file is empty? For now, just assume it's good.
            skipped_count += 1
            continue
            
        # 2. Download if missing
        print(f"   [{i+1}/{len(registry)}] Restoring missing file: {item['name'][:50]}...")
        
        try:
            results = client.get(dataset_id, limit=DOWNLOAD_ROWS_LIMIT)
            df = pd.DataFrame.from_records(results)
            
            if df.empty:
                print("      ⚠️ Warning: Dataset is empty on server.")
                
            df.to_csv(csv_path, index=False)
            downloaded_count += 1
            
        except Exception as e:
            print(f"       Failed to restore {dataset_id}: {e}")

    print("-" * 50)
    print(" Sync Complete!")
    print(f"   - Existing files skipped: {skipped_count}")
    print(f"   - Missing files downloaded: {downloaded_count}")
    print("-" * 50)

if __name__ == "__main__":
    sync_data_from_registry()