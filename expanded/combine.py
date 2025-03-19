import json
import csv
import sqlite3
import os
from glob import glob

# Paths (adjust if needed)
output_file = "TAGBench_expanded.json"

# Load Reasoning queries from final_movies_data.json
with open("final_movies_data.json") as f:
    reasoning_data = json.load(f)

reasoning_entries = []
for entry in reasoning_data:
    reasoning_entries.append({
        "db": "rotten_tomatoes",
        "question": entry["question"],
        "question_type": "Reasoning",
        "answer": entry["answer"]
    })

# Load Knowledge queries from human_filtered_train.json and run SQL for answers
with open("human_filtered_train.json") as f:
    knowledge_data = json.load(f)

knowledge_entries = []

# Assumes SQLite databases are located in a folder called 'databases' parallel to 'expanded'
db_folder = "/home/asimbiswal/TAG-Bench/bird_train/train/train_databases"

for entry in knowledge_data:
    db_id = entry["db_id"]
    db_path = os.path.join(db_folder + f"/{db_id}", f"{db_id}.sqlite")
    # print(db_path)
    question = entry["new_question"]
    sql = entry["SQL"]

    if not os.path.exists(db_path):
        continue  # Skip if database doesn't exist

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()

        if results:
            if len(results) == 1:
                # One row only
                row = results[0]
                answer = row[0] if len(row) == 1 else row
            else:
                # Multiple rows — return all rows as list of values or rows
                answer = [row[0] if len(row) == 1 else row for row in results]
        else:
            continue  # Skip if results is 

        knowledge_entries.append({
            "db": db_id,
            "question": question,
            "question_type": "Knowledge",
            "answer": answer
        })
        # Skip if result is None (no answer)
    except Exception:
        continue  # Skip entries with SQL errors

# Load Knowledge queries from CSV files
csv_files = glob("*.csv")

for csv_file in csv_files:
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        db_col, question_col = header[0], header[1]

        for row in reader:
            db_val = row[0] + "_masked"
            question_val = row[1]
            knowledge_entries.append({
                "db": db_val,
                "question": question_val,
                "question_type": "Knowledge",
                "answer": "TODO"
            })

# Combine all entries
all_entries = knowledge_entries + reasoning_entries

# Save to TAGBench_expanded.json
with open(output_file, "w") as f:
    json.dump(all_entries, f, indent=2)

print(f"Saved {len(all_entries)} queries to {output_file}")
