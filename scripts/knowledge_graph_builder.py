import json
import os
from tqdm import tqdm
import re

# --- Configuration ---
SOURCE_FILE = 'confluence_dump.jsonl'
OUTPUT_FILE = 'knowledge_graph.json'

def extract_entities():
    """
    Reads the source Confluence dump and extracts key entities and relationships
    to build a simple knowledge graph.
    """
    print(f"Starting knowledge graph extraction from '{SOURCE_FILE}'...")
    knowledge_graph = {}

    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: Source file not found at '{SOURCE_FILE}'")
        return

    for line in tqdm(lines, desc="Extracting Entities"):
        try:
            entry = json.loads(line)
            title = entry.get('title', 'Untitled Document')
            doc_id = entry.get('id', title)
            text = entry.get('text', '')

            # Extract Jira tickets (e.g., GM-12345)
            jira_tickets = list(set(re.findall(r'\b([A-Z]+-\d+)\b', text + title)))

            # Extract links to Google Workspace documents
            google_sheets = list(set(re.findall(r'https?://docs\.google\.com/spreadsheets/d/[a-zA-Z0-9_-]+', text)))
            google_docs = list(set(re.findall(r'https?://docs\.google\.com/document/d/[a-zA-Z0-9_-]+', text)))
            google_slides = list(set(re.findall(r'https?://docs\.google\.com/presentation/d/[a-zA-Z0-9_-]+', text)))

            knowledge_graph[doc_id] = {
                "title": title,
                "jira_tickets": jira_tickets,
                "related_docs": [],
                "google_sheets": google_sheets,
                "google_docs": google_docs,
                "google_slides": google_slides
            }

        except json.JSONDecodeError:
            continue

    all_titles = {data['title']: doc_id for doc_id, data in knowledge_graph.items()}
    for doc_id, data in tqdm(knowledge_graph.items(), desc="Finding Relationships"):
        text_content = ""
        for line in lines:
            # A more robust way to find the matching line for the doc_id
            try:
                if json.loads(line).get('id') == doc_id or json.loads(line).get('title') == data['title']:
                     text_content = json.loads(line).get('text', '')
                     break
            except json.JSONDecodeError:
                continue
        
        for title, related_doc_id in all_titles.items():
            if title in text_content and doc_id != related_doc_id:
                data['related_docs'].append(related_doc_id)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(knowledge_graph, f_out, indent=2)

    print(f"\nKnowledge graph extraction complete. Output saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    extract_entities()

