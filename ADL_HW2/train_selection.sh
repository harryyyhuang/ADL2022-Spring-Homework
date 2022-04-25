python3 run_swag_no_trainer.py \
        --train_file ./cache/train.csv \
        --validation_file ./cache/valid.csv \
        --max_length 512 \
        --model_name_or_path bert-base-chinese \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_train_epochs 10 \
        --output_dir ./ckpt/multi_choice/