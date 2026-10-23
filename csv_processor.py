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

def load_replies(path: str | Path) -> pd.DataFrame: #Load and validate the CSV
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
        raise ValueError("Duplicate IDs found in the input CSV")

    if df["their_reply"].isna().any():
        raise ValueError("Some rows have missing customer replies.")

    return df

def process_replies_stream(
    input_path: str | Path,
    output_path: str = "output/analyzed_replies.csv",
    bypass_cache: bool = False,
):
    """Generator that yields (processed_row, current_index, total_count) as each row finishes."""
    df = load_replies(input_path)
    results = []
    total = len(df)

    for index, row in enumerate(df.to_dict(orient="records"), start=1): #made a dictionary of each row and we are traversing on it, start=1 because we want count from 1 not 0
        try:
            result = reply_graph.invoke({
                "row": {key: "" if pd.isna(value) else str(value) for key, value in row.items()} ,
                "attempts": 0,
                "bypass_cache": bypass_cache,
            })
            print("result===================",result)
            print()
            print()
            print()
            print()

            analysis = result.get("analysis")
            print("analysis====================",analysis)
            print()
            print()
            print()
            print()
            print()
            print()

            if analysis is None:
                raise ValueError("Graph returned no analysis.")

            analysis_data = (
                analysis.model_dump(mode="json")
                if hasattr(analysis, "model_dump")
                else analysis
            )
            print("analysis_data=====================",analysis_data)
            print()
            print()
            print()
            print()
            print()
            print()


            processed_row = {
                **row,
                **analysis_data,
            }
            print("processed_row==================",processed_row)
            print()
            print()
            print()
            print()
            print()
            print()
            print()

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
        yield processed_row, index, total #wait karne ki jagah it sends these value to it's caller

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"\nFinished. Output saved to: {output.resolve()}")


def process_replies(
    input_path: str | Path,
    output_path: str = "output/analyzed_replies.csv",
    progress_callback=None,
    bypass_cache: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Run each reply through LangGraph and export the results."""
    results = []
    reset_stats() #change all stats to 0 in SQlite 
    for processed_row, index, total in process_replies_stream(
        input_path, output_path, bypass_cache=bypass_cache
    ):
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
    parser.add_argument("input", nargs="?", default="replies2.csv")
    parser.add_argument(
        "--output",
        default="output/analyzed_replies.csv",
    )
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Bypass SQLite cache and run fresh classifications",
    )

    args = parser.parse_args()
    process_replies(args.input, args.output, bypass_cache=args.bypass_cache)