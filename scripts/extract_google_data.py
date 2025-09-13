import os.path
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

# --- Configuration ---
# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]
KNOWLEDGE_GRAPH_FILE = 'knowledge_graph.json'
OUTPUT_FILE = 'gworkspace_knowledge.json'
CREDENTIALS_FILE = 'credentials.json' # Download from Google Cloud Console
TOKEN_FILE = 'token.json'

def authenticate_google():
    """Handles Google API authentication."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                print(f"Error: '{CREDENTIALS_FILE}' not found.")
                print("Please download it from your Google Cloud project's credentials page.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

def get_all_gworkspace_urls(kg_path):
    """Extracts all unique Google Workspace URLs from the knowledge graph."""
    try:
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
    except FileNotFoundError: return {}
    
    urls = {
        'sheets': set(),
        'docs': set(),
        'slides': set() # Slides API is more complex, focusing on text extraction
    }
    for data in kg.values():
        for url in data.get('google_sheets', []): urls['sheets'].add(url)
        for url in data.get('google_docs', []): urls['docs'].add(url)
        # Slides extraction can be added here if needed
    return {k: list(v) for k, v in urls.items()}

def extract_doc_content(service, doc_id):
    """Extracts text from a Google Doc."""
    try:
        doc = service.documents().get(documentId=doc_id).execute()
        content = ""
        for element in doc.get('body').get('content'):
            if 'paragraph' in element:
                for pe in element.get('paragraph').get('elements'):
                    if 'textRun' in pe:
                        content += pe.get('textRun').get('content')
        return content
    except HttpError as e:
        return f"Error extracting content: {e}"

def extract_sheet_content(service, sheet_id):
    """Extracts all cell data from all tabs in a Google Sheet."""
    try:
        sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = sheet_metadata.get('sheets', '')
        content = {}
        for sheet in sheets:
            title = sheet.get('properties', {}).get('title', 'Untitled Sheet')
            result = service.spreadsheets().values().get(
                spreadsheetId=sheet_id, range=title).execute()
            values = result.get('values', [])
            content[title] = values # Store as a list of lists (rows)
        return content
    except HttpError as e:
        return f"Error extracting content: {e}"

def extract_gworkspace_data():
    creds = authenticate_google()
    if not creds:
        print("Authentication failed. Exiting.")
        return

    urls = get_all_gworkspace_urls(KNOWLEDGE_GRAPH_FILE)
    if not any(urls.values()):
        print("No Google Workspace URLs found in the knowledge graph."); return

    gworkspace_knowledge = {}
    
    # Process Google Docs
    if urls['docs']:
        docs_service = build('docs', 'v1', credentials=creds)
        for url in tqdm(urls['docs'], desc="Extracting Google Docs"):
            doc_id = url.split('/d/')[1].split('/')[0]
            gworkspace_knowledge[url] = extract_doc_content(docs_service, doc_id)

    # Process Google Sheets
    if urls['sheets']:
        sheets_service = build('sheets', 'v4', credentials=creds)
        for url in tqdm(urls['sheets'], desc="Extracting Google Sheets"):
            sheet_id = url.split('/d/')[1].split('/')[0]
            gworkspace_knowledge[url] = extract_sheet_content(sheets_service, sheet_id)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(gworkspace_knowledge, f_out, indent=2)

    print(f"\nGoogle Workspace extraction complete. Knowledge saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    extract_gworkspace_data()
