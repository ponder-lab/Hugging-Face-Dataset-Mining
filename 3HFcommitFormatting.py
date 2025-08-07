#!/usr/bin/env python3
import argparse
import json
import pandas as pd

#Formats the commits into a readable spreadsheet

def flatten_commits(input_csv: str, output_csv: str):
    # Load the dataset
    df = pd.read_csv(input_csv)

    records = []
    for _, row in df.iterrows():
        dataset_id = row.get('datasetId') or row.get('DatasetID')
        commits_raw = row.get('commits')

        # Skip if there are no commits or not a JSON string
        if not isinstance(commits_raw, str):
            continue

        try:
            commits_list = json.loads(commits_raw)
        except json.JSONDecodeError:
            continue

        for c in commits_list:
            records.append({
                'DatasetID':          dataset_id,
                'CommitId':           c.get('commit_id'),
                'Authors':            c.get('authors'),
                'Date':               c.get('created_at'),
                'Log message':        c.get('title'),
                'message':            c.get('message'),
                'formatted_title':    c.get('formatted_title'),
                'formatted_message':  c.get('formatted_message'),
            })

    # Build DataFrame and write out
    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten JSON-encoded commits in a CSV into individual rows"
    )
    parser.add_argument("input_csv",  help="Path to original CSV (with a 'commits' column)")
    parser.add_argument("output_csv", help="Path for the flattened output CSV")
    args = parser.parse_args()
    flatten_commits(args.input_csv, args.output_csv)
