import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd


def eval(args):
    df = pd.read_csv(args.df_path)
    output_dir = args.output_dir

    grouped = df.groupby("Query type")
    latencies = defaultdict(list)
    corrects = defaultdict(list)
    for group_name, group_df in grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]

            with open(f"{output_dir}/query_{qid}.json") as f:
                data = json.load(f)
                latencies[group_name].append(data["latency"])
                if data.get("error", None):
                    corrects[group_name].append(False)
                else:
                    corrects[group_name].append(data["prediction"] == data["answer"])

    kr_grouped = df.groupby("Knowledge/Reasoning Type")
    for group_name, group_df in kr_grouped:
        for _, row in group_df.iterrows():
            qid = row["Query ID"]
            if row["Query type"] != "Aggregation":
                with open(f"{output_dir}/query_{qid}.json") as f:
                    data = json.load(f)
                    latencies[group_name].append(data["latency"])
                    if data.get("error", None):
                        corrects[group_name].append(False)
                    else:
                        corrects[group_name].append(data["prediction"] == data["answer"])

    for k, v in latencies.items():
        print(f"Printing stats for {k}")
        print(f"Mean latency: {np.mean(v):.2f}")
        print(f"Avg. correct: {np.mean(corrects[k]):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", help="Path to the CSV file containing the queries")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the output files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval(args)
