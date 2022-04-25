accelerate launch run_qa_no_trainer.py \
  --train_file ./cache/qa/train.json \
  --validation_file ./cache/qa/valid.json \
  --model_name_or_path hfl/chinese-macbert-large \
  --max_seq_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./ckpt/qa/
