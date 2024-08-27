import argparse
import os

import pandas as pd
from lotus.models import E5Model

from tag.utils import row_to_str


def embed_df(args):
    e5_model = E5Model()
    df_path = os.path.join("../pandas_dfs", args.db_name, f"{args.table_name}.csv")
    df = pd.read_csv(df_path)
    serialized_rows = []

    for _, row in df.iterrows():
        serialized_rows.append(f"passage: {row_to_str(row)}")

    print(f"Embedded DataFrame: {args.table_name}")
    print(f"Number of rows: {len(serialized_rows)}")

    print(serialized_rows[0])
    index_dir = os.path.join("../indexes", args.db_name, args.table_name)
    e5_model.index(serialized_rows, index_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Embed a DataFrame")
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--table_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed_df(args)
