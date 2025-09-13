import os
import json
from jira import JIRA
from tqdm import tqdm
import requests
import warnings

# --- Configuration ---
JIRA_SERVER_URL = "https://work.greyorange.com/jira"
JIRA_PAT = "Enter your PAT here" # Your PAT
# NEW: Control SSL verification with an environment variable for security. Defaults to True.
JIRA_SSL_VERIFY = False


KNOWLEDGE_GRAPH_FILE = 'knowledge_graph.json'
OUTPUT_FILE = 'jira_export_deep.json'
ATTACHMENTS_DIR = 'jira_attachments'
MAX_RECURSION_DEPTH = 2

def get_all_ticket_ids(kg_path):
    """Extracts all unique JIRA ticket IDs from the knowledge graph."""
    try:
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
        all_ids = set(ticket_id for data in kg.values() for ticket_id in data.get('jira_tickets', []))
        return list(all_ids)
    except FileNotFoundError:
        print(f"Error: Knowledge graph file '{kg_path}' not found.")
        return []

def download_attachments(issue, jira, issue_key):
    """Downloads all attachments for a given Jira issue using the authenticated session."""
    attachment_metadata = []
    if not hasattr(issue.fields, 'attachment') or not issue.fields.attachment:
        return attachment_metadata

    ticket_attachment_dir = os.path.join(ATTACHMENTS_DIR, issue_key)
    os.makedirs(ticket_attachment_dir, exist_ok=True)

    for attachment in issue.fields.attachment:
        try:
            # Use the authenticated session from the jira object to download
            response = jira._session.get(attachment.content, stream=True)
            response.raise_for_status()
            file_path = os.path.join(ticket_attachment_dir, attachment.filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            attachment_metadata.append({'filename': attachment.filename, 'local_path': file_path})
        except Exception as e:
            print(f"\nWarning: Failed to download attachment '{attachment.filename}'. Error: {e}")
    return attachment_metadata

def get_issue_details_recursively(issue_key, jira, processed_tickets, current_depth=0):
    """Recursively fetches details for a Jira issue and its linked issues/subtasks."""
    if current_depth > MAX_RECURSION_DEPTH or issue_key in processed_tickets:
        return None
    processed_tickets.add(issue_key)
    try:
        issue = jira.issue(issue_key)
        details = {
            'summary': issue.fields.summary, 'description': issue.fields.description,
            'status': issue.fields.status.name,
            #'attachments': download_attachments(issue, jira, issue_key), # Pass the main jira object
            'subtasks': [], 'linked_issues': []
        }
        for subtask in issue.fields.subtasks:
            sub_details = get_issue_details_recursively(subtask.key, jira, processed_tickets, current_depth + 1)
            if sub_details: details['subtasks'].append({subtask.key: sub_details})
        for link in issue.fields.issuelinks:
            linked_issue = getattr(link, 'outwardIssue', getattr(link, 'inwardIssue', None))
            if linked_issue:
                link_details = get_issue_details_recursively(linked_issue.key, jira, processed_tickets, current_depth + 1)
                if link_details: details['linked_issues'].append({'type': link.type.name, linked_issue.key: link_details})
        return details
    except Exception as e:
        print(f"\nWarning: Could not fetch data for ticket {issue_key}. Error: {e}")
        return None

def export_jira_data():
    """Connects to Jira using PAT and exports data."""
    try:
        # --- UPDATED: Connection using PAT and configurable SSL verification ---
        if not JIRA_SSL_VERIFY:
            print("WARNING: JIRA_SSL_VERIFY is set to False. This disables SSL certificate verification and is insecure.")
            print("This should only be used in trusted, controlled environments.")
            warnings.filterwarnings('ignore', 'Unverified HTTPS request')
        
        jira_options = {
            'server': JIRA_SERVER_URL,
            'verify': JIRA_SSL_VERIFY
        }
        jira = JIRA(options=jira_options, token_auth=JIRA_PAT)
        print("Successfully connected to Jira!")

    except Exception as e:
        print(f"Error connecting to Jira: {e}"); return
        
    root_ticket_ids = get_all_ticket_ids(KNOWLEDGE_GRAPH_FILE)
    if not root_ticket_ids:
        print("No JIRA tickets found in knowledge graph to process.")
        return

    jira_data, processed_tickets = {}, set()
    os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
    for ticket_id in tqdm(root_ticket_ids, desc="Exporting Jira Data & Attachments"):
        if ticket_id not in processed_tickets:
            jira_data[ticket_id] = get_issue_details_recursively(ticket_id, jira, processed_tickets)
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: json.dump(jira_data, f, indent=2)
    print(f"\nJira export complete.")

if __name__ == "__main__":
    export_jira_data()


