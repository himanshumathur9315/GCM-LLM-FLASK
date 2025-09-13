import google.generativeai as genai
import json
import os
from tqdm import tqdm
import time

# --- Configuration ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY not set."); exit()

SOURCE_FILE = 'confluence_dump.jsonl'
KNOWLEDGE_GRAPH_FILE = 'knowledge_graph.json'
JIRA_EXPORT_FILE = 'jira_export_deep.json'
ATTACHMENTS_KNOWLEDGE_FILE = 'attachments_knowledge.json'
GWORKSPACE_KNOWLEDGE_FILE = 'gworkspace_knowledge.json'
OUTPUT_FILE = 'generated_sft_data_complete.jsonl'
MODEL_NAME = 'gemini-1.5-flash-latest'

# --- DEFINITIVE, FULLY-DETAILED PROMPT ---
PROMPT_TEMPLATE = """You are an expert AI Data Architect creating a definitive fine-tuning dataset. Synthesize information from a Confluence document, its related Jira tickets, textual descriptions of its attachments, AND the content of linked Google Workspace files.

Based ONLY on the provided context, generate a JSON list with a comprehensive and diverse set of instruction-output pairs. Be exhaustive.

**Generate a diverse mix of the following ELEVEN types of tasks:**

**1. Summarization Task:**
   - "instruction": Ask for a high-level summary of the main Confluence document, referencing its title.
   - "output": A concise summary of the document's purpose.

**2. Key Information Extraction Task:**
   - "instruction": Ask to extract specific, important entities (e.g., components, goals) from the main document, referencing its title.
   - "output": A structured list (using markdown) of the requested information.

**3. Procedural / How-To Task:**
   - "instruction": If a process is described in the main document, ask for the steps to perform that task, referencing the title.
   - "output": A numbered or bulleted list of the steps.

**4. Relationship / Dependency Task:**
   - "instruction": If dependencies are mentioned in the main document, ask to identify them, referencing the title.
   - "output": A list of other systems or components the subject interacts with.

**5. Persona-Based Extraction Task:**
   - "instruction": Frame a question about the main document from the perspective of a specific role (e.g., QA Engineer, Product Manager), referencing the title.
   - "output": An answer tailored to that role's concerns (e.g., testing focus, business impact).

**6. Negative Finding / Boundary Definition Task:**
   - "instruction": Ask if the main document contains information on a plausible but *absent* topic, referencing the title.
   - "output": A confirmation that the topic is not discussed in the provided document.

**7. Metadata Querying Task:**
   - "instruction": Ask for a specific piece of metadata found in the main document, like the owner, status, or a JIRA ticket.
   - "output": The direct answer to the metadata question.

**8. Multi-Document Reasoning Task:**
   - "instruction": Frame a question that requires information from the main document AND references a related Confluence document title from the Knowledge Graph Context.
   - "output": An answer that synthesizes information, acknowledging both sources.

**9. Jira Synthesis Task:**
   - "instruction": Frame a question that requires synthesizing information between the main Confluence document and the details of a specific Jira ticket (e.g., its description, acceptance criteria, or comments).
   - "output": An answer that explicitly combines information from both sources, citing both by title/ID.

**10. Attachment Synthesis Task:**
   - "instruction": Frame a question that requires information from the textual analysis of a specific attachment, referencing the attachment's filename.
   - "output": An answer based on the analyzed content of the attachment, citing the source file.

**11. Google Workspace Synthesis Task:**
   - "instruction": Frame a question that requires information from a linked Google Doc or Sheet, referencing its URL.
   - "output": An answer based on the extracted content of the Google Workspace file, citing the source URL.

**Rules:**
- ALWAYS cite your sources (document title, Jira ID, filename, URL) in the instruction and output.
- Base your answers STRICTLY on the provided text. Do not invent.
- Be exhaustive.

**Confluence Document:**
---
**Title:** {title}
**Text:** {context}

**Knowledge Graph Context:**
---
{kg_context}

**Detailed Jira Data:**
---
{jira_context}

**Textual Analysis of Attachments:**
---
{attachment_context}

**Extracted Content from Linked Google Workspace Files:**
---
{gworkspace_context}
"""

def load_json_file(filepath, desc):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {desc} file '{filepath}' not found."); return {}

def get_attachment_knowledge(ticket_data, attachment_db):
    knowledge = {}
    if isinstance(ticket_data, dict):
        for att in ticket_data.get('attachments', []):
            path = att.get('local_path')
            if path in attachment_db: knowledge[att.get('filename')] = attachment_db[path]
        for value in ticket_data.values(): knowledge.update(get_attachment_knowledge(value, attachment_db))
    elif isinstance(ticket_data, list):
        # --- FIXED: Corrected variable name from 'data' to 'ticket_data' ---
        for item in ticket_data: knowledge.update(get_attachment_knowledge(item, attachment_db))
    return knowledge

def generate_data():
    kg = load_json_file(KNOWLEDGE_GRAPH_FILE, "Knowledge graph")
    jira_data = load_json_file(JIRA_EXPORT_FILE, "Jira export")
    attachment_knowledge = load_json_file(ATTACHMENTS_KNOWLEDGE_FILE, "Attachment knowledge")
    gworkspace_knowledge = load_json_file(GWORKSPACE_KNOWLEDGE_FILE, "Google Workspace knowledge")
    if not kg: print("Knowledge graph is essential."); return

    model = genai.GenerativeModel(MODEL_NAME)
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f_in, open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Generating Final Instructions"):
            try:
                entry = json.loads(line)
                doc_id = entry.get('id', entry.get('title'))
                kg_entry = kg.get(doc_id, {})
                
                # --- Build Comprehensive Context for the Prompt ---
                jira_context = {tid: jira_data.get(tid) for tid in kg_entry.get('jira_tickets', [])}
                attachment_context = get_attachment_knowledge(jira_context, attachment_knowledge)
                gworkspace_context = {
                    url: gworkspace_knowledge.get(url) for type in ['google_sheets', 'google_docs', 'google_slides'] 
                    for url in kg_entry.get(type, []) if url in gworkspace_knowledge
                }

                prompt = PROMPT_TEMPLATE.format(
                    title=entry.get('title', 'Untitled'), context=entry.get('text', ''),
                    kg_context=json.dumps(kg_entry, indent=2),
                    jira_context=json.dumps(jira_context, indent=2),
                    attachment_context=json.dumps(attachment_context, indent=2),
                    gworkspace_context=json.dumps(gworkspace_context, indent=2)
                )

                # --- API Call and Write to File ---
                response = model.generate_content(prompt)
                json_start = response.text.find('[')
                json_end = response.text.rfind(']') + 1
                if json_start != -1:
                    generated_pairs = json.loads(response.text[json_start:json_end])
                    for pair in generated_pairs:
                        if 'instruction' in pair and 'output' in pair:
                            f_out.write(json.dumps(pair) + '\n')
            except Exception as e:
                print(f"\nError processing entry {doc_id}. Error: {e}"); continue
    print(f"\nDefinitive data generation complete.")

if __name__ == "__main__":
    generate_data()

