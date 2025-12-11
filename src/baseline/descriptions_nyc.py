"""
File: src/baseline/descriptions_nyc.py
Description: 
    Implements the logic for AutoDDG-NYC (Domain Customization).
    Contains specialized prompts and generation functions for NYC-specific descriptions.
"""

import json
from baseline.llm_client import call_llm
from baseline.descriptions_autoddg import sample_rows

# --- Helper: Convert data to JSON string safely ---
def _safe_json_block(obj, label="profile"):
    """
    Dumps a dictionary to a JSON string.
    Truncates the string if it exceeds 8000 characters to save tokens/cost.
    """
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) > 8000:
        return s[:8000] + "...(truncated)"
    return s

# --- 1. Custom Prompt: User-Focused (NYC Resident Context) ---
UFD_NYC_PROMPT = """You are a data analyst for NYC Open Data.
Your goal is to write a dataset description specifically for New York City residents, researchers, and policymakers.

### Dataset Metadata
- **Agency**: {agency}
- **Topic**: {topic}
- **Category**: {category}

### Data Context
- **Sample Data**:
{sample}

- **Content Statistics**:
{content_profile}

- **Semantic Meaning**:
{semantic_profile}

### Instructions
Write a "User-Focused Description" (6-10 sentences).
1. **Contextualize**: Explicitly mention that this data comes from **{agency}**.
2. **Explain Content**: Mention specific NYC entities found in the data (e.g., specific Boroughs, Community Districts, NYPD Precincts, Zip Codes).
3. **Utility**: Explain how a NYC citizen can use this? (e.g., "Check restaurant grades in Queens", "Analyze 311 noise complaints").
4. **Tone**: Helpful, civic-minded, and professional.

Output ONLY the description text.
"""

# --- 2. Custom Prompt: Search-Focused (SEO Context) ---
SFD_NYC_PROMPT = """You are an SEO expert for the NYC Open Data Portal.
Generate a "Search-Focused Description" to help users find this dataset.

Input Description:
{ufd_nyc}

Metadata:
- Agency: {agency}
- Category: {category}

Instructions:
Generate the following sections in plain text:

Dataset Overview:
A 2-3 sentence summary highlighting the agency ({agency}) and key subjects.

Key NYC Entities & Locations:
List specific NYC geographic units or entities found (e.g., "Manhattan", "Tax Lots", "Subway Lines").

Potential Use Cases:
List 3 specific questions this data can answer for New Yorkers.

Keywords:
5-8 comma-separated keywords (include abbreviations like "TLC", "DOB", "NYPD" if relevant).
"""

# --- Main Function: Generate UFD_NYC ---
def generate_ufd_nyc(csv_path, content_profile, semantic_profile, topic, metadata):
    """
    Generates the NYC-specific User-Focused Description using the custom prompt.
    """
    # Extract agency/category from metadata; use defaults if missing
    agency = metadata.get("agency", "NYC Agency")
    category = metadata.get("category", "General")
    title = metadata.get("name", "NYC Dataset")
    
    # Prepare data inputs
    sample = sample_rows(csv_path)
    content_json = _safe_json_block(content_profile)
    semantic_json = _safe_json_block(semantic_profile)
    
    # Format the prompt
    prompt = UFD_NYC_PROMPT.format(
        title=title, agency=agency, category=category, topic=topic,
        sample=sample, content_profile=content_json, semantic_profile=semantic_json
    )
    
    # Call the LLM
    return call_llm(prompt, temperature=0.3)

# --- Main Function: Generate SFD_NYC ---
def generate_sfd_nyc(ufd_nyc, metadata):
    """
    Generates the NYC-specific Search-Focused Description based on the UFD.
    """
    agency = metadata.get("agency", "NYC Agency")
    category = metadata.get("category", "General")
    
    prompt = SFD_NYC_PROMPT.format(ufd_nyc=ufd_nyc, agency=agency, category=category)
    return call_llm(prompt, temperature=0.3)