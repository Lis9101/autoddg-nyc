"""
File: src/pipeline_test.py
Description: 
    This script executes an end-to-end test of the data processing pipeline.
    It performs two main steps:
    1. Connects to the NYC Open Data (Socrata) API to fetch a sample dataset.
    2. Sends a data preview to the Google Gemini API to verify description generation.
    
    Usage: python src/pipeline_test.py
"""

import os
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load environment variables
load_dotenv()
SOCRATA_TOKEN = os.getenv("SOCRATA_APP_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Check if Keys exist
if not SOCRATA_TOKEN or not GEMINI_KEY:
    print("Error: Missing Token or Key in .env file, please check!")
    exit()

# 2. Configure Gemini AI
print("Configuring Gemini AI...")
genai.configure(api_key=GEMINI_KEY)
# Using the latest flash model which we confirmed is available
model = genai.GenerativeModel('gemini-2.0-flash')

def run_test_pipeline():
    print("-" * 40)
    print("Starting End-to-End Pipeline Test")
    print("-" * 40)

    # --- Stage 1: Download data from NYC Open Data ---
    print("\n1️⃣  Stage 1: Downloading data from Socrata...")
    client = Socrata("data.cityofnewyork.us", SOCRATA_TOKEN)
    
    # We select an interesting dataset: "Central Park Squirrel Census"
    # ID: vfnx-vebw
    dataset_id = "vfnx-vebw" 
    
    try:
        # Get Metadata
        metadata = client.get_metadata(dataset_id)
        dataset_name = metadata['name']
        print(f"   -> Found dataset: {dataset_name}")
        
        # Download data (Download only first 5 rows for testing)
        results = client.get(dataset_id, limit=5)
        df = pd.DataFrame.from_records(results)
        
        # Convert to string format, ready to feed to AI
        # Send only column names and first 3 rows to save tokens
        data_preview = df.head(3).to_markdown(index=False)
        print(f"   -> Data downloaded successfully! Column count: {len(df.columns)}")
        
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return

    # --- Stage 2: Generate description using Gemini ---
    print("\n2️⃣  Stage 2: Sending to Gemini for description generation...")
    
    # Construct Prompt
    prompt = f"""
    You are a data analyst. I will give you a sample of a dataset from NYC Open Data.
    
    Dataset Name: {dataset_name}
    
    Data Sample:
    {data_preview}
    
    Task: Write a 2-sentence description of what this dataset appears to be about.
    """
    
    try:
        # Call API
        response = model.generate_content(prompt)
        
        print("\n Gemini Generated Description:")
        print("=" * 40)
        print(response.text)
        print("=" * 40)
        print("\n Test Success! Your pipeline is working!")
        
    except Exception as e:
        print(f" Stage 2 failed: {e}")

if __name__ == "__main__":
    run_test_pipeline()