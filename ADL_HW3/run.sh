python3 summay_predict.py \
        --model_name_or_path summary \
        --validation_file ${1} \
        --source_prefix "summarize: " \
        --source_prefix "summarize: " \
        --text_column "maintext" \
        --seed 601 \
        --output_file ${2}