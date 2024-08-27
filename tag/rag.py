import argparse
import json
import os
import re
import time

import pandas as pd
from lotus.models import E5Model, OpenAIModel

from tag.utils import IndexMerger, row_to_str

# Taken from STaRK - https://arxiv.org/pdf/2404.13207
RERANK_PROMPT_TEMPLATE = (
    "You are a helpful assistant that examines if a row "
    "satisfies a given query and assign a score from 0.0 to 1.0. "
    "If the row does not satisfy the query, the score should be 0.0. "
    "If there exists explicit and strong evidence supporting that row "
    "satisfies the query, the score should be 1.0. If partial evidence or weak "
    "evidence exists, the score should be between 0.0 and 1.0.\n"
    'Here is the query:\n"{query}"\n'
    "Here is the information about the row:\n{row_str}\n\n"
    "Please score the row based on how well it satisfies the query. "
    "ONLY output the floating point score WITHOUT anything else. "
    "Output: The numeric score of this row is: "
)


def run_row(args, query_row):
    index_merger = db_used_to_index_merger[query_row["DB used"]]
    question = query_row["Query"]
    try:
        answer = eval(query_row["Answer"])
    except Exception:
        answer = query_row["Answer"]

    tic = time.time()
    if args.rerank:
        results = index_merger(question, args.ret_k)
        for result in results:
            row_str = row_to_str(result.row)
            prompt = RERANK_PROMPT_TEMPLATE.format(query=question, row_str=row_str)
            reranker_score = lm([[{"role": "user", "content": prompt}]])[0]
            reranker_score = float(re.findall(r"[-+]?\d*\.\d+|\d+", reranker_score)[0])
            result.reranker_score = reranker_score

        results = sorted(results, key=lambda x: x.reranker_score, reverse=True)
    else:
        results = index_merger(question, args.ret_k)

    user_instruction = ""
    for i, result in enumerate(results):
        user_instruction += f"Data Point {i+1}\n{row_to_str(result.row)}\n\n"

    user_instruction += f"Question: {question}"
    if query_row["Query type"] == "Aggregation":
        system_instruction = (
            "You will be given a list of data points and a question. Use the data points to answer the question. "
            "If a value is a string, it must be enclosed in double quotes."
        )
    else:
        system_instruction = (
            "You will be given a list of data points and a question. Use the data points to answer the question. "
            "Your answer must be a list of values that is evaluatable in Python. Respond in the format [value1, value2, ..., valueN]."
            "If you are unable to answer the question, respond with []. Respond with only the list of values and nothing else. "
            "If a value is a string, it must be enclosed in double quotes."
        )

    messages = [[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction}]]
    prediction = lm(messages)[0]
    latency = time.time() - tic

    try:
        prediction = eval(prediction)
    except Exception:
        print(f"Error evaluating prediction: {prediction}")
    if not isinstance(answer, list):
        answer = [answer]

    return {
        "prediction": prediction,
        "answer": answer,
        "query_id": query_row["Query ID"],
        "latency": latency,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--df_path", default="../tag_queries.csv")
    parser.add_argument("--ret_k", type=int, default=5)
    parser.add_argument("--output_dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    queries_df = pd.read_csv(args.df_path)

    lm = OpenAIModel(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct", api_base="http://localhost:8000/v1", provider="vllm"
    )
    rm = E5Model()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    db_used_to_index_merger = {
        "california_schools": IndexMerger(
            [("california_schools", "frpm"), ("california_schools", "satscores"), ("california_schools", "schools")], rm
        ),
        "codebase_community": IndexMerger(
            [
                ("codebase_community", "badges"),
                ("codebase_community", "comments"),
                ("codebase_community", "postHistory"),
                ("codebase_community", "postLinks"),
                ("codebase_community", "posts"),
                ("codebase_community", "tags"),
                ("codebase_community", "users"),
                ("codebase_community", "votes"),
            ],
            rm,
        ),
        "debit_card_specializing": IndexMerger(
            [
                ("debit_card_specializing", "customers"),
                ("debit_card_specializing", "gasstations"),
                ("debit_card_specializing", "products"),
                ("debit_card_specializing", "sqlite_sequence"),
                ("debit_card_specializing", "transactions_1k"),
                ("debit_card_specializing", "yearmonth"),
            ],
            rm,
        ),
        "european_football_2": IndexMerger(
            [
                ("european_football_2", "Country"),
                ("european_football_2", "League"),
                ("european_football_2", "Match"),
                ("european_football_2", "Player"),
                ("european_football_2", "Player_Attributes"),
                ("european_football_2", "sqlite_sequence"),
                ("european_football_2", "Team"),
                ("european_football_2", "Team_Attributes"),
            ],
            rm,
        ),
        "formula_1": IndexMerger(
            [
                ("formula_1", "circuits"),
                ("formula_1", "constructorResults"),
                ("formula_1", "constructors"),
                ("formula_1", "constructorStandings"),
                ("formula_1", "drivers"),
                ("formula_1", "driverStandings"),
                ("formula_1", "lapTimes"),
                ("formula_1", "pitStops"),
                ("formula_1", "qualifying"),
                ("formula_1", "races"),
                ("formula_1", "results"),
                ("formula_1", "seasons"),
                ("formula_1", "sqlite_sequence"),
                ("formula_1", "status"),
            ],
            rm,
        ),
    }

    for _, query_row in queries_df.iterrows():
        output = run_row(args, query_row)

        print(output)
        if args.output_dir:
            with open(os.path.join(args.output_dir, f"query_{query_row['Query ID']}.json"), "w+") as f:
                json.dump(output, f)
