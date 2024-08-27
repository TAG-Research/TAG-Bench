"""
Code heavily taken from BIRD
https://arxiv.org/pdf/2305.03111
https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/gpt_request.py
"""

import argparse
import sqlite3

import pandas as pd
from tqdm import tqdm


def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = "".join(f"{column.rjust(width)} " for column, width in zip(column_names, widths))
    # Print the values
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + "\n" + rows
    return final_output


def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    """
    :param root_place:
    :param db_name:
    :return:
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == "sqlite_sequence":
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ["order", "by", "group"]:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                num_rows, cur_table, num_rows, rows_prompt
            )
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt


def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + "\n" + question_prompt
    else:
        result_prompt = knowledge_prompt + "\n" + pattern_prompt_kg + "\n" + question_prompt

    return result_prompt


def cot_wizard():
    cot = "\nGenerate the SQL after thinking step by step: "

    return cot


def few_shot():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. referring to external knowledge, we need to filter singers 'by year' - 'birth_year' > 27; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE year - birth_year > 27;</s>"

    one_shot_demo = ini_table + "\n" + ini_prompt + "\n" + ini_cot_result

    return one_shot_demo


def few_shot_no_kg():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    age  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge:\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. 'older than 27' refers to age > 27 in SQL; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE age > 27;</s>"

    one_shot_demo = ini_table + "\n" + ini_prompt + "\n" + ini_cot_result

    return one_shot_demo


def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)  # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    # combined_prompts = schema_prompt + "\n\n" + comment_prompt + cot_wizard() + "\nSELECT "
    combined_prompts = schema_prompt + "\n\n" + comment_prompt + "\nSELECT "
    return combined_prompts


def collect_all_prompts(db_path_list, question_list, knowledge_list=None):
    """
    :param db_path: str
    :param question_list: []
    """
    all_prompts = []
    for i, question in tqdm(enumerate(question_list)):
        print("--------------------- processing {}th question ---------------------".format(i))
        print("the question is: {}".format(question))

        if knowledge_list:
            cur_prompt = generate_combined_prompts_one(
                db_path=db_path_list[i], question=question, knowledge=knowledge_list[i]
            )
        else:
            cur_prompt = generate_combined_prompts_one(db_path=db_path_list[i], question=question)

        all_prompts.append(cur_prompt)
        print(cur_prompt)

    return all_prompts


def decouple_question_schema(tag_queries_df, db_root_path, naivetag: bool = False):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for _, row in tag_queries_df.iterrows():
        if naivetag:
            question_list.append(row["(TAG baseline) Text2SQL Input"])
        else:
            question_list.append(row["Query"])
        cur_db_path = db_root_path + row["DB used"] + "/" + row["DB used"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(row.get("evidence", None))

    return question_list, db_path_list, knowledge_list


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--tag_queries_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument("--output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("-n", "--naive_tag", action="store_true")
    args = args_parser.parse_args()

    tag_queries_df = pd.read_csv(args.tag_queries_path)

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        tag_queries_df=tag_queries_df, db_root_path=args.db_root_path, naivetag=args.naive_tag
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    all_prompts = collect_all_prompts(
        db_path_list=db_path_list,
        question_list=question_list,
        knowledge_list=knowledge_list if args.use_knowledge == "True" else None,
    )

    # Add all prompts list as a column in the tag_queries_df
    if args.naive_tag:
        tag_queries_df["naivetag_text2sql_prompt"] = all_prompts
    else:
        tag_queries_df["text2sql_prompt"] = all_prompts
    tag_queries_df.to_csv(args.output_path, index=False)
