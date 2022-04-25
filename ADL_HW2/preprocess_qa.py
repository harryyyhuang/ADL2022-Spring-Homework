from argparse import ArgumentParser
from pathlib import Path
import json
import logging
import csv


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)



def main(args):
    contexts_path = args.data_dir / "context.json"
    contexts = json.loads(contexts_path.read_text())
    
    for split in ["train", "valid"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")

        output_path = args.output_dir / f"{split}.json"
        custom_json = []


        for i, data in enumerate(dataset):

            one_data = {}
            one_data["id"] = data["id"]
            one_data["title"] = "custom"
            one_data["context"] = contexts[data["relevant"]]
            one_data["question"] = data["question"]
            one_data["answers"] = {"text": [data["answer"]["text"]], "answer_start": [data["answer"]["start"]]}


            custom_json.append(one_data)

        output_json = {"data": custom_json}
        output_path.write_text(json.dumps(output_json, indent = 2, ensure_ascii=False))

        logging.info(f"{split} qa preprocess saved at {str(output_path.resolve())}")





def parser_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file",
        default="./cache/qa/"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
