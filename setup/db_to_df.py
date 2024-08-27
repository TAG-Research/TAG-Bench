import argparse
import os
import sqlite3

import pandas as pd


def convert_db_to_df(args):
    conn = sqlite3.connect(f"../dev_folder/dev_databases/{args.db_name}/{args.db_name}.sqlite")
    # Get the list of table names in the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Iterate over each table and read the data into a DataFrame
    table_name_to_pandas_df = {}
    for table in tables:
        table_name = table[0]
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        table_name_to_pandas_df[table_name] = df

    # Store dataframes
    os.makedirs(f"../pandas_dfs/{args.db_name}", exist_ok=True)
    for table_name, df in table_name_to_pandas_df.items():
        df.to_csv(f"../pandas_dfs/{args.db_name}/{table_name}.csv", index=False)
        print(f"Saved {table_name} to ../pandas_dfs/{args.db_name}/{table_name}.csv")

    conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a BIRD database to a DataFrame")
    parser.add_argument("--db_name", type=str, help="Path to the BIRD database", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_db_to_df(args)
