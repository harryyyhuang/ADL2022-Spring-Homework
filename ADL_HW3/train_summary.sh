python3 summary_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --text_column "maintext" \
    --summary_column "title" \
    --num_beams 3 \
    --seed 601 \
    --with_tracking \
    --num_train_epochs 20 \
    --output_dir ./summary_beam3 \