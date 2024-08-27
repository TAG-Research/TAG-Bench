wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip -O dev.zip

# Unzip folders
unzip dev.zip
rm dev.zip
mv dev_20240627 ../dev_folder
unzip ../dev_folder/dev_databases.zip -d ../dev_folder

# Convert databases to dataframes
databases=("california_schools" "debit_card_specializing" "codebase_community" "formula_1" "european_football_2")

for db_name in "${databases[@]}"
do
    python db_to_df.py --db_name "$db_name"
done