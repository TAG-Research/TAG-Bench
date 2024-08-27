tag_queries_path='../tag_queries.csv'
db_root_path='../dev_folder/dev_databases/'
use_knowledge='True'
not_use_knowledge='False'
mode='dev'
cot='True'
no_cot='False'

# Overwrite the tag_queries.csv file to have prompts
output_path='../tag_queries.csv'

echo 'generate prompts without knowledge or CoT'
python -u get_text2sql_prompts.py --db_root_path ${db_root_path} --mode ${mode} \
    --tag_queries_path ${tag_queries_path} --output_path ${output_path} --use_knowledge ${not_use_knowledge} \
    --chain_of_thought ${no_cot}

echo 'generate prompts for naive tag'
python -u get_text2sql_prompts.py -n --db_root_path ${db_root_path} --mode ${mode} \
    --tag_queries_path ${tag_queries_path} --output_path ${output_path} --use_knowledge ${not_use_knowledge} \
    --chain_of_thought ${no_cot}