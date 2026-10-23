## Making python env
python -m venv .venv 
source .venv/bin/activate 

## Install Requirements 
pip install langgraph groq pandas pydantic python-dotenv gradio pytest

## Run Web UI
python app.py