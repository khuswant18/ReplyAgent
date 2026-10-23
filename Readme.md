# Sales Reply Triage Agent

An AI-powered tool that analyzes sales email replies, classifies prospect intent, and recommends the next action using Groq, LangGraph, and Pydantic.

## Features

- **Reply Classification:** Categorizes replies as `yes`, `no`, `future`, or `needs_review`.
- **Intent Extraction:** Identifies interest, pricing questions, referrals, objections, and follow-ups.
- **Smart Recommendations:** Suggests next actions and drafts replies when appropriate.
- **Safety Checks:** Flags explicit opt-outs and uncertain responses for appropriate handling.
- **SQLite Caching:** Reuses successful classifications to reduce repeated API calls.
- **CSV Processing:** Upload, analyze, view, and download results through a Gradio interface.

## Tech Stack

Python, Gradio, LangGraph, Groq API, Pydantic, Pandas, SQLite.

## Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/khuswant18/ReplyAgent.git
cd ReplyAgent
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model
```

Start the application:

```bash
python app.py
```

Open the local URL displayed in your terminal.

## Safety

The agent does not automatically send emails. Review flagged responses and generated drafts before taking action.

## Deployment

The application can be deployed as a web service on Render. Configure `GROQ_API_KEY` and `GROQ_MODEL` as environment variables in the service settings.