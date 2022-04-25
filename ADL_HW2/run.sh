python3 preprocess_choice_test.py --data_json ${2} --context_json ${1}
python3 predict_multiple_choice.py
if [ ! -d predict/ ]; then
    mkdir -p predict/
fi
python3 predict_qa.py
python3 post_process_qa.py --output_file ${3}