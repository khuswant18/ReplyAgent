from pathlib import Path 
import pandas as pd 
from graph import reply_graph
from cache import reset_stats, get_stats

wanted_col = [
    "id",
    "received_at",
    "sender_name",
    "sender_title",
    "company",
    "our_message",
    "their_reply",
]

def load_replies(path: str | Path) -> pd.DataFrame:
    """Load and validate the input CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {exc}")

    if df.empty:
        raise ValueError("The input CSV is empty.")

    missing = set(wanted_col) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in the input CSV.")

    if df["their_reply"].isna().any():
        raise ValueError("Some rows have missing customer replies.")

    return df

def process_replies_stream(
    input_path: str | Path,
    output_path: str = "output/analyzed_replies.csv",
):
    """Generator that yields (processed_row, current_index, total_count) as each row finishes."""
    df = load_replies(input_path)
    results = []
    total = len(df)

    for index, row in enumerate(df.to_dict(orient="records"), start=1):
        try:
            result = reply_graph.invoke({
                "row": {
                    key: "" if pd.isna(value) else str(value)
                    for key, value in row.items()
                },
                "attempts": 0,
            })

            analysis = result.get("analysis")

            if analysis is None:
                raise ValueError("Graph returned no analysis.")

            analysis_data = (
                analysis.model_dump(mode="json")
                if hasattr(analysis, "model_dump")
                else analysis
            )

            processed_row = {
                **row,
                **analysis_data,
                "processing_status": "success",
                "processing_error": "",
            }

        except Exception as exc:
            processed_row = {
                **row,
                "disposition": "needs_review",
                "intent": "ambiguous",
                "next_action": "human_review",
                "needs_human_review": True,
                "processing_status": "failed",
                "processing_error": type(exc).__name__,
            }

        results.append(processed_row)
        yield processed_row, index, total

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"\nFinished. Output saved to: {output.resolve()}")


def process_replies(
    input_path: str | Path,
    output_path: str = "output/analyzed_replies.csv",
    progress_callback=None,
) -> tuple[pd.DataFrame, str]:
    """Run each reply through LangGraph and export the results."""
    results = []
    reset_stats()
    for processed_row, index, total in process_replies_stream(input_path, output_path):
        if progress_callback:
            progress_callback(index / total, desc=f"Processing reply {index}/{total}")
        else:
            print(f"Processing reply {index}/{total}")
        results.append(processed_row)

    stats = get_stats()
    print("\n--- Classification Usage ---")
    print(f"Groq API calls: {stats['api_calls']}")
    print(f"Cache hits:     {stats['cache_hits']}")
    print(f"Cache misses:   {stats['cache_misses']}")

    return pd.DataFrame(results), str(Path(output_path).resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="replies.csv")
    parser.add_argument(
        "--output",
        default="output/analyzed_replies.csv",
    )

    args = parser.parse_args()
    process_replies(args.input, args.output)