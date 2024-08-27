import argparse
import json
import os
import re
import sqlite3
import time

import pandas as pd
from lotus.models import OpenAIModel


def run_row(query_row):
    text2sql_prompt = query_row["text2sql_prompt"]
    db_name = query_row["DB used"]
    raw_answer = lm([[{"role": "user", "content": text2sql_prompt}]])[0]
    sql_statements = re.findall(r"```sql\n(.*?)\n```", raw_answer, re.DOTALL)
    if not sql_statements:
        sql_statements = re.findall(r"```\n(.*?)\n```", raw_answer, re.DOTALL)
    if not sql_statements:
        sql_statements = [raw_answer]

    last_sql_statement = sql_statements[-1]
    try:
        try:
            answer = eval(query_row["Answer"])
        except Exception:
            answer = query_row["Answer"]

        conn = sqlite3.connect(f"../dev_folder/dev_databases/{db_name}/{db_name}.sqlite")
        cursor = conn.cursor()
        cursor.execute(last_sql_statement)
        raw_results = cursor.fetchall()
        predictions = [res[0] for res in raw_results]

        if not isinstance(answer, list):
            predictions = predictions[0] if predictions else None

        return {
            "query_id": query_row["Query ID"],
            "sql_statement": last_sql_statement,
            "prediction": predictions,
            "answer": answer,
        }

    except Exception as e:
        return {
            "error": f"Error running SQL statement: {last_sql_statement}\n{e}",
            "query_id": query_row["Query ID"],
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", default="../tag_queries.csv", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    queries_df = pd.read_csv(args.df_path)
    lm = OpenAIModel(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_base="http://localhost:8000/v1",
        provider="vllm",
        max_tokens=512,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for _, query_row in queries_df.iterrows():
        tic = time.time()
        output = run_row(query_row)
        latency = time.time() - tic

        output["latency"] = latency
        print(output)
        if args.output_dir:
            with open(os.path.join(args.output_dir, f"query_{query_row['Query ID']}.json"), "w+") as f:
                json.dump(output, f)
