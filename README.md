# AI Agent For Email


This is readme to create an useful AI Agent for your gmail account. It requires some prerequestie. It uses local LLM and make sure complete snapshot sent to the user so they are not missing any steps.\

First run the requirements so all the required packages are installed.
pip install -r requirements.txt

You need google developer account so the user communicate with gmail client
Gmail Developer Setup (credentials.json)
Google Cloud Console:
Go to https://console.cloud.google.com
Create a new Project.
Enable Gmail API:
In “APIs & Services” dashboard, click “Enable APIs”, search for “Gmail API”, and enable it.
OAuth Consent Screen:
Configure the OAuth consent (select “External” for personal use).
Fill out details (app name, user support email, etc.).

Create OAuth Client ID:
Go to “Credentials” (sidebar).
“+ Create credentials” → “OAuth client ID”
Select “Desktop App”
Give it a name (e.g. gmail_agent_for_finance)
Click “Create” and then “Download JSON”
Save as credentials.json in the same directory as the script.

First Script Run:

Script will launch the OAuth flow (local browser opens for login/consent).
After approval, token.json will save your auth tokens for future runs.

How to Run
Install dependencies:

pip install -r requirements.txt
Place the downloaded credentials.json in the script directory.

Run the script:
python gmail_agent_for_finance.py
First time: Complete Google login in browser.

Daily summaries are emaild to the user
