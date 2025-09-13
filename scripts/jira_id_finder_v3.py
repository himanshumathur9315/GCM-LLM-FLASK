import os
import warnings
from jira import JIRA

# --- Configuration ---
# This script uses the same environment variables as your main exporter script.
JIRA_SERVER_URL = "https://work.greyorange.com/jira"
gJIRA_PAT = "JIRA_PAT" # Your PAT
# NEW: Control SSL verification with an environment variable for security. Defaults to True.
JIRA_SSL_VERIFY = False

def find_custom_fields():
    """
    Connects to Jira, inspects a single issue, and prints its custom fields and IDs using a more robust method.
    """
    ticket_id = input("Enter a Jira ticket ID that has the custom fields you're looking for (e.g., GM-238534): ")
    if not ticket_id:
        print("No ticket ID provided. Exiting.")
        return

    try:
        if not JIRA_SSL_VERIFY:
            warnings.filterwarnings('ignore', 'Unverified HTTPS request')
        
        jira_options = {
            'server': JIRA_SERVER_URL,
            'verify': JIRA_SSL_VERIFY
        }
        jira = JIRA(options=jira_options, token_auth=JIRA_PAT)
        print("\nSuccessfully connected to Jira!")

    except Exception as e:
        print(f"Error connecting to Jira: {e}"); return

    try:
        # --- FIXED: Use a more reliable two-step method ---
        # 1. Get all fields from the Jira instance to build a reliable map.
        print("Fetching all available field definitions from Jira...")
        all_fields = jira.fields()
        field_name_map = {field['id']: field['name'] for field in all_fields}
        print("Successfully built field map.")

        # 2. Get the specific issue.
        issue = jira.issue(ticket_id)
        print(f"\nFound the following custom fields with values on ticket {ticket_id}:\n")
        print("-" * 50)
        print(f"{'Field Name':<30} | {'Field ID':<20}")
        print("-" * 50)

        # 3. Iterate through the issue's fields and look them up in our map.
        # We check issue.raw['fields'] which contains all fields, even if they are empty.
        for field_id, field_value in issue.raw['fields'].items():
            if field_id.startswith('customfield_'):
                # We only print the field if it has a value on this ticket, to reduce noise.
                if field_value:
                    field_name = field_name_map.get(field_id, "Unknown Custom Field")
                    print(f"{field_name:<30} | {field_id:<20}")

        print("-" * 50)
        print("\nFind the fields you need in the list above and copy their IDs into the CUSTOM_FIELD_MAPPING in your 'export_jira_data.py' script.")

    except Exception as e:
        print(f"Error fetching issue {ticket_id}. Please ensure the ticket exists and you have permission to view it.")
        print(f"Details: {e}")

if __name__ == "__main__":
    find_custom_fields()

