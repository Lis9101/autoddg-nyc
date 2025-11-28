"""
File: src/check_models.py
Description: 
    This utility script lists all Google Gemini models available to your specific API Key.
    Use this to verify which model versions (e.g., 'gemini-1.5-flash', 'gemini-2.0-flash') 
    you have access to if you encounter '404 Not Found' errors.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Querying all models supported by your API Key...\n")

try:
    # 2. List all models
    count = 0
    for m in genai.list_models():
        # Only show models that support text generation
        if 'generateContent' in m.supported_generation_methods:
            print(f"Available model: {m.name}")
            count += 1
            
    if count == 0:
        print("  No models found that support generateContent.")
    else:
        print(f"\nFound {count} available models!")

except Exception as e:
    print(f" Query failed: {e}")