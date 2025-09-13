import google.generativeai as genai
import os
import json
from tqdm import tqdm
import time
from pathlib import Path

# --- Configuration ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set."); exit()

JIRA_EXPORT_FILE = 'jira_export_deep.json'
OUTPUT_FILE = 'attachments_knowledge.json'
MODEL_NAME = 'gemini-1.5-pro-latest'

# --- ENHANCED: Type-Specific Prompts and Supported Formats ---
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.webp']
SUPPORTED_DOC_FORMATS = ['.pdf', '.docx'] # <-- Added .docx
SUPPORTED_DATA_FORMATS = ['.csv', '.tsv', '.xlsx', '.xls'] # <-- Added Excel formats

# Generic prompt for images
IMAGE_PROMPT = """You are an expert AI data analyst. Your task is to analyze the provided image and generate a concise, accurate, and detailed textual description of its content.
- If it's a **diagram/flowchart**, describe the components, connections, and the process it illustrates.
- If it's a **screenshot**, extract all readable text and describe the visual elements of the UI.
- If it's **handwritten notes**, transcribe the text and describe the layout."""

# Specific prompt for text-based documents (PDF, DOCX)
DOC_PROMPT = "You are an expert document analyst. Provide a comprehensive summary of the key sections, findings, and conclusions from the attached document. Extract any critical data points, tables, or action items."

# Specific prompt for tabular data (CSV, XLSX)
DATA_PROMPT = "You are an expert data analyst. Analyze the provided spreadsheet file. Describe its structure (including any multiple sheets/tabs), identify the key columns, and provide a high-level summary of the data it contains. Do not output the raw data; summarize its meaning and any obvious trends."


def analyze_attachments():
    try:
        with open(JIRA_EXPORT_FILE, 'r', encoding='utf-8') as f:
            jira_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Jira export file '{JIRA_EXPORT_FILE}' not found."); return

    attachments_to_process = []
    def find_attachments(data):
        if isinstance(data, dict):
            if 'attachments' in data: attachments_to_process.extend(data['attachments'])
            for value in data.values(): find_attachments(value)
        elif isinstance(data, list):
            for item in data: find_attachments(item)
    find_attachments(jira_data)
    
    attachments_knowledge = {}
    model = genai.GenerativeModel(MODEL_NAME)

    for attachment in tqdm(attachments_to_process, desc="Analyzing Attachments"):
        local_path = Path(attachment.get('local_path', ''))
        file_extension = local_path.suffix.lower()

        # --- ENHANCED: File Type Handling Logic ---
        prompt_to_use = None
        if file_extension in SUPPORTED_IMAGE_FORMATS:
            prompt_to_use = IMAGE_PROMPT
        elif file_extension in SUPPORTED_DOC_FORMATS:
            prompt_to_use = DOC_PROMPT
        elif file_extension in SUPPORTED_DATA_FORMATS:
            prompt_to_use = DATA_PROMPT
        else:
            # Gracefully skip unsupported files with more informative message
            unsupported_msg = f"Analysis skipped: File format '{file_extension}' is not directly supported."
            if file_extension == '.dwg':
                unsupported_msg += " For CAD files, please attach a PDF or PNG export for analysis."
            attachments_knowledge[str(local_path)] = unsupported_msg
            continue

        if local_path.exists() and local_path.stat().st_size > 0:
            try:
                print(f"\nAnalyzing {local_path}...")
                uploaded_file = genai.upload_file(path=local_path)
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2); uploaded_file = genai.get_file(uploaded_file.name)
                if uploaded_file.state.name == "FAILED": raise ValueError(f"File processing failed for {local_path.name}")
                
                # Use the selected prompt
                response = model.generate_content([prompt_to_use, uploaded_file])
                attachments_knowledge[str(local_path)] = response.text
                genai.delete_file(uploaded_file.name)

            except Exception as e:
                print(f"\nError analyzing file {local_path}. Error: {e}")
                attachments_knowledge[str(local_path)] = f"Error during analysis: {e}"

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: json.dump(attachments_knowledge, f, indent=2)
    print(f"\nVisual analysis complete. Knowledge saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    analyze_attachments()

