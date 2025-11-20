"""
File: src/data_collector.py
Description: 
    This script executes the "Bulk Data Collection" phase.
    
    Updates in this version:
    Sorts by popularity (page_views) to get the most relevant datasets first.
    
    Usage: python src/data_collector.py
"""

import os
import json
import time
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv

# --- Configuration ---
TARGET_DATASET_COUNT = 2000   # Test with 20, then change to 2000
DOWNLOAD_ROWS_LIMIT = 100   
DATA_DIR = "data"
METADATA_FILE = os.path.join(DATA_DIR, "metadata_registry.json")

# 1. Load environment variables
load_dotenv()
SOCRATA_TOKEN = os.getenv("SOCRATA_APP_TOKEN")

if not SOCRATA_TOKEN:
    print("Error: SOCRATA_APP_TOKEN not found in .env file.")
    exit()

def collect_data():
    print("-" * 50)
    print(f" Starting Bulk Data Collection (Target: {TARGET_DATASET_COUNT})")
    print(f"Strategy: Most Popular (Page Views Last Month)")
    print("-" * 50)

    client = Socrata("data.cityofnewyork.us", SOCRATA_TOKEN)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Load existing registry
    registry = []
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            print(f" Loaded existing registry with {len(registry)} datasets.")
        except:
            registry = []

    collected_count = len(registry)
    page_size = 50
    offset = 0

    # Main Loop
    while collected_count < TARGET_DATASET_COUNT:
        print(f"\n Scanning catalog (Offset: {offset})...")
        
        try:
            # Get a batch of datasets
            # We order by popularity to ensure our sample is "representative" of usage
            search_results = client.datasets(
                limit=page_size, 
                offset=offset, 
                order="page_views_last_month DESC" 
            )
        except Exception as e:
            # This catches the "Expected 50 got 11" error at the end of the list
            print(f"  End of search results or API limitation reached: {e}")
            break

        if not search_results:
            print("  No more datasets returned.")
            break

        processed_in_batch = 0

        for item in search_results:
            if collected_count >= TARGET_DATASET_COUNT:
                break
            
            # The ID is nested in 'resource' for search results
            dataset_id = item.get('resource', {}).get('id', item.get('id'))
            
            # Skip if already downloaded
            if any(d['id'] == dataset_id for d in registry):
                continue

            try:
                # === CRITICAL FIX ===
                # The search result 'item' often lacks 'columns'.
                # We MUST fetch the specific metadata for this ID to check if it's a table.
                full_metadata = client.get_metadata(dataset_id)
                
                # 1. Filter: Must be Tabular (has columns)
                if 'columns' not in full_metadata:
                    # This is likely a map, a link, or a file, not a table
                    continue
                
                name = full_metadata['name']
                print(f"    [{collected_count + 1}/{TARGET_DATASET_COUNT}] Downloading: {name[:50]}...")

                # 2. Download Data Sample
                results = client.get(dataset_id, limit=DOWNLOAD_ROWS_LIMIT)
                df = pd.DataFrame.from_records(results)
                
                if df.empty:
                    print("        Skipped (Empty dataset)")
                    continue

                # 3. Save CSV
                csv_filename = f"{dataset_id}.csv"
                csv_path = os.path.join(DATA_DIR, csv_filename)
                df.to_csv(csv_path, index=False)

                # 4. Record Metadata
                meta_entry = {
                    "id": dataset_id,
                    "name": name,
                    "description": full_metadata.get("description", ""),
                    "category": full_metadata.get("category", "Uncategorized"),
                    "agency": full_metadata.get("attribution", "Unknown"),
                    "columns": [col['name'] for col in full_metadata.get("columns", [])], 
                    "local_path": csv_path,
                    "sample_row_count": len(df),
                    "page_views": item.get('page_views_last_month', 0) # Good for analysis
                }
                registry.append(meta_entry)
                collected_count += 1
                
                # Save continuously
                with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(registry, f, indent=4)
                    
            except Exception as e:
                # If fetch fails, just skip this dataset and move to next
                # print(f"      Error processing {dataset_id}: {e}") # Uncomment to debug
                pass
            
            processed_in_batch += 1

        # If we went through a whole batch and found nothing valid, we still need to advance
        offset += page_size
        time.sleep(0.5) 

    print("-" * 50)
    print(f" Collection Complete!")
    print(f" Total Datasets Collected: {collected_count}")
    print("-" * 50)

if __name__ == "__main__":
    collect_data()