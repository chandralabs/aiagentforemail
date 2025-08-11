"""
Gmail Email Summarizer & Financial Assistant - Scheduled Daily at 8PM
Powered by llama.cpp + HuggingFace GGUF models
- Finds deals/coupons, bill due dates, payment history, savings advice
- Auto-selects CPU/GPU/MPS at runtime
- Sends a report email at 8:00 PM every day
"""

import os, base64, time
from pathlib import Path
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from llama_cpp import Llama
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from huggingface_hub import hf_hub_download
import schedule

MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
GGUF_FILE = None
MAX_CONTEXT, MAX_TOKENS, TEMP = 8192, 1000, 0.1
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send"
]
MODEL_CACHE_DIR = Path("models")
YOUR_EMAIL = "your_email@gmail.com"
REPORT_SUBJECT = "Gmail Agent Report"

# Download model if needed and initialize Llama
def get_model_path(model_id, filename=None):
    return hf_hub_download(repo_id=model_id, filename=filename,
                           local_dir=MODEL_CACHE_DIR, local_dir_use_symlinks=False)

def detect_hardware():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "Metal", 999
        elif torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if mem_gb >= 16: return "CUDA", 999
            elif mem_gb >= 8: return "CUDA", 40
            elif mem_gb >= 4: return "CUDA", 20
            else: return "CUDA", 10
    except ImportError:
        pass
    return "CPU", 0

def load_model():
    path = get_model_path(MODEL_REPO, GGUF_FILE)
    backend, n_gpu_layers = detect_hardware()
    print(f"[LLM] Backend: {backend}, n_gpu_layers={n_gpu_layers}")
    return Llama(model_path=path, n_ctx=MAX_CONTEXT, n_threads=os.cpu_count(), n_gpu_layers=n_gpu_layers)

# Gmail OAuth authentication and message fetching
def parse_parts(parts):
    txt, html = "", ""
    for part in parts:
        mime = part.get("mimeType")
        body = part.get("body", {})
        data = body.get("data")
        if mime == 'multipart/alternative':
            t, h = parse_parts(part.get("parts", []))
            txt += t; html += h
        elif mime == 'text/plain' and data:
            txt += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        elif mime == 'text/html' and data:
            html += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return txt, html

def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_emails():
    service = get_gmail_service()
    messages = service.users().messages().list(userId="me", q="newer_than:1d category:primary").execute()
    results = []
    for idx, meta in enumerate(messages.get("messages", []), 1):
        msg = service.users().messages().get(userId="me", id=meta["id"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        subject = headers.get("Subject", "No Subject")
        sender = headers.get("From", "Unknown Sender")
        txt, html = parse_parts([msg["payload"]])
        body = txt if txt.strip() else BeautifulSoup(html, "html.parser").get_text()
        results.append((subject, sender, body))
        print(f"[GMAIL] Retrieved #{idx}")
    return results

# Summarizing each message using LLM
def summarize_email(email_body, llm):
    prompt = f"""
You are a financial-savvy personal assistant. Extract:
1. Deals/coupons with codes & expiration
2. Bill deadlines (dates, amounts, providers)
3. Past payment history
4. Money-saving tips
5. Summary + IMPORTANT/UNIMPORTANT tag
EMAIL:
{email_body}
"""
    resp = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMP, stop=["</s>"])
    return resp["choices"][0]["text"].strip()

def daily_overview(important_emails, llm):
    prompt = f"""Given these important emails:
1. To-Do list (urgent)
2. Active deals/coupons
3. Bill history
4. Savings recommendations
5. Short narrative summary
{important_emails}
"""
    resp = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMP, stop=["</s>"])
    return resp["choices"][0]["text"].strip()

# Format the report in HTML for email
def format_html_report(report_text):
    # Here you can parse `report_text` further for more structure if needed
    return f"""
    <html>
    <body style="font-family:Arial,sans-serif;">
    <h2 style="background:#2979ff;color:#fff;padding:10px;">Gmail Agent Report</h2>
    <div style="border:1px solid #ddd;padding:16px;">
      {report_text.replace('\\n', '<br>')}
    </div>
    <footer style="margin-top:20px;font-size:smaller;color:#888;">
      <hr>
      <em>Generated automatically at 8:00 PM.</em>
    </footer>
    </body>
    </html>
    """

# Email sending using Gmail API
def send_report_via_gmail(report_text):
    html_report = format_html_report(report_text)
    service = get_gmail_service()
    message = MIMEMultipart("alternative")
    message["to"] = test@testmail.com
    message["subject"] = REPORT_SUBJECT
    message.attach(MIMEText(report_text, "plain"))
    message.attach(MIMEText(html_report, "html"))
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"[EMAIL SENT] to {test@testmail.com}")

# Task function to run daily
def run_daily_job():
    print("[JOB] Running Gmail Agent...")
    llm_model = load_model()
    emails = fetch_emails()
    summaries = []
    for subj, sender, body in emails:
        if not body.strip(): continue
        summ = summarize_email(body, llm_model)
        summaries.append((subj, sender, summ))
    important = [s for s in summaries if "IMPORTANT" in s[2]]
    all_important_text = "\n\n".join([x[2] for x in important])
    report = daily_overview(all_important_text, llm_model)
    send_report_via_gmail(report)

# Scheduler keeps the script running, triggers job at 8PM daily
if __name__ == "__main__":
    schedule.every().day.at("20:00").do(run_daily_job)
    print("[SCHEDULER] Waiting for 8 PM daily run...")
    while True:
        schedule.run_pending()
        time.sleep(30)

