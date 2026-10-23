# Sales Reply Triage Agent

An AI tool that classifies inbound sales email replies, extracts prospect intent, and recommends next actions using LangGraph, Groq, and Pydantic.

## Features

- **Reply Classification:** Classifies replies into `yes`, `no`, `future`, or `needs_review`.
- **Intent Extraction:** Detects intents like `interested`, `pricing_question`, `request_for_examples`, `referral`, `out_of_office`, `unsubscribe`, and `wrong_contact`.
- **Next Action & Drafts:** Recommends follow-up actions (`offer_call`, `send_examples`, `answer_pricing`, etc.) and generates suggested email drafts.
- **Opt-Out Safety:** Regex-based rules detect explicit unsubscribe requests and suppress contact without calling the LLM.
- **Dual Processing Modes:**
  - **Process Replies:** Uses SQLite caching to return instant results for previously analyzed replies and save API calls.
  - **Run Fresh:** Bypasses cache reads and writes to force fresh LLM analysis.
- **Web UI & CLI:** Upload CSVs via a Gradio web interface or process them from the command line.

## Tech Stack

- Python 3.11+
- LangGraph
- Groq API
- Pydantic v2
- Gradio
- Pandas
- SQLite

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/khuswant18/ReplyAgent.git
cd ReplyAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
PORT=7860
```

3. Run the Gradio web interface:

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

4. Run via CLI:

```bash
# Process with caching
python csv_processor.py replies.csv --output output/analyzed_replies.csv

# Process fresh (bypass cache)
python csv_processor.py replies.csv --bypass-cache
```

## CSV Format

The input CSV requires the following columns:

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `received_at` | Timestamp of the reply |
| `sender_name` | Prospect's name |
| `sender_title` | Prospect's job title |
| `company` | Company name |
| `our_message` | Outreach message sent |
| `their_reply` | Prospect's reply text |

## Testing

Run tests with pytest:

```bash
pytest
```

## Safety

The agent does not send emails automatically. All generated drafts and actions require human review before reaching out to prospects.