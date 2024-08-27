import argparse
import json
import os
import re
import sqlite3
import time

import pandas as pd
from lotus.models import OpenAIModel

from tag.utils import row_to_str


def run_row(query_row):
    text2sql_prompt = query_row["naivetag_text2sql_prompt"]
    question = query_row["Query"]
    db_name = query_row["DB used"]
    messages = [[{"role": "user", "content": text2sql_prompt}]]
    raw_answer = lm(messages)[0]
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
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(last_sql_statement)
        raw_results = cursor.fetchall()
        row_dicts = []
        for result in raw_results:
            row_dicts.append({col: result[col] for col in result.keys()})

        user_instruction = ""

        for i, row in enumerate(row_dicts):
            user_instruction += f"Data Point {i+1}\n{row_to_str(row)}\n\n"

        user_instruction += f"Question: {question}"
        system_instruction = (
            "You will be given a list of data points and a question. Use the data points to answer the question. "
            "Your answer must be a list of values that is evaluatable in Python. Respond in the format [value1, value2, ..., valueN]."
            "If you are unable to answer the question, respond with []. Respond with only the list of values and nothing else. "
            "If a value is a string, it must be enclosed in double quotes."
        )

        messages = [[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction}]]
        if lm.count_tokens(messages[0]) > 130000:
            print("Context length exceeded")
            return {
                "query_id": query_row["Query ID"],
                "prediction": "None",
                "answer": answer,
                "sql_statement": last_sql_statement,
            }

        prediction = lm(messages)[0]

        try:
            prediction = eval(prediction)
        except Exception:
            prediction = prediction

        if not isinstance(answer, list):
            prediction = prediction[0] if prediction else None

        return {
            "query_id": query_row["Query ID"],
            "prediction": prediction,
            "answer": answer,
            "sql_statement": last_sql_statement,
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
