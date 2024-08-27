for folder in ../pandas_dfs/*; do
    db_name=$(basename -- "$folder")
    for file in $folder/*; do
        table_name=$(basename -- "$file");
        table_name="${table_name%.*}"

        echo "Embedding $db_name/$table_name"
        python embed_df.py --db_name "$db_name" --table_name "$table_name"
    done
done