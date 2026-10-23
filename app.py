import os
from pathlib import Path
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from csv_processor import process_replies

load_dotenv()


def run_triage(file, progress=gr.Progress(track_tqdm=True)):

    if file is None:
        raise gr.Error("upload a CSV file.")

    file_path = file.name if hasattr(file, "name") else str(file)

    if not os.path.exists(file_path):
        raise gr.Error("Uploaded file not found")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = str(output_dir / "analyzed_replies.csv")

    try:
        result_df, saved_path = process_replies(
            input_path=file_path,
            output_path=out_csv_path,
            progress_callback=progress,
        )
    except ValueError as ve:
        raise gr.Error(f"Validation Error: {ve}")
    except Exception as exc:
        raise gr.Error(f"Processing Error: {exc}")

    total = len(result_df)
    yes_count = int((result_df["disposition"] == "yes").sum())
    no_count = int((result_df["disposition"] == "no").sum())
    future_count = int((result_df["disposition"] == "future").sum())
    review_count = int(
        (
            (result_df["disposition"] == "needs_review")
            | (result_df["needs_human_review"] == True)
        ).sum()
    )

    display_columns = [
        "sender_name",
        "company",
        "their_reply",
        "disposition",
        "intent",
        "priority",
        "next_action",
        "needs_human_review",
    ]
    rename_map = {
        "sender_name": "Sender",
        "company": "Company",
        "their_reply": "Their Reply",
        "disposition": "Disposition",
        "intent": "Intent",
        "priority": "Priority",
        "next_action": "Next Action",
        "needs_human_review": "Needs Review",
    }
    available_cols = [c for c in display_columns if c in result_df.columns]
    table_df = result_df[available_cols].rename(columns=rename_map)

    return (
        total,
        yes_count,
        no_count,
        future_count,
        review_count,
        table_df,
        gr.update(value=saved_path, visible=True),
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Sales Reply Triage Agent") as demo:
        gr.Markdown("""# Sales Reply Triage Agent Upload customer outreach replies in CSV format to classify intent, 
                    identify opt-outs, and recommend next actions using LangGraph and Groq.""")

        with gr.Row(): #in one row we have upload and process
            with gr.Column(scale=2):
                csv_file = gr.File(
                    label="Upload Sales Replies CSV",
                    file_types=[".csv"],
                    file_count="single",
                )
            with gr.Column(scale=1):
                process_btn = gr.Button(
                    "Process Replies",
                    variant="primary",
                    size="lg",
                )
                download_file = gr.File(
                    label="Download Analyzed CSV",
                    visible=False,
                    interactive=False,
                )

        gr.Markdown("### Summary")
        with gr.Row():
            total_metric = gr.Number(
                label="Total Replies",
                value=0,
                interactive=False,
            )
            yes_metric = gr.Number(
                label="Interested (Yes)",
                value=0,
                interactive=False,
            )
            no_metric = gr.Number(
                label="Not Interested (No)",
                value=0,
                interactive=False,
            )
            future_metric = gr.Number(
                label="Future Follow-up",
                value=0,
                interactive=False,
            )
            review_metric = gr.Number(
                label="Needs Review",
                value=0,
                interactive=False,
            )

        gr.Markdown("### Results Table")
        results_table = gr.DataFrame(
            headers=[
                "Sender",
                "Company",
                "Their Reply",
                "Disposition",
                "Intent",
                "Priority",
                "Next Action",
                "Needs Review",
            ],
            interactive=False,
            wrap=True,
        )

        process_btn.click(
            fn=run_triage,
            inputs=[csv_file],
            outputs=[
                total_metric,
                yes_metric,
                no_metric,
                future_metric,
                review_metric,
                results_table,
                download_file,
            ],
        )

    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="blue",neutral_hue="slate",)
    server_port = int(os.environ.get("PORT", "7860"))
    server_name = os.environ.get("HOST", "0.0.0.0")

    app = create_app()
    app.launch(theme=theme,server_name=server_name,server_port=server_port,)
