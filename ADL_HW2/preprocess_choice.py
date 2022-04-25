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

        output_path = args.output_dir / f"{split}.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["","video-id","fold-ind","startphrase","sent1","sent2","gold-source","ending0","ending1","ending2","ending3","label"])
            for i, data in enumerate(dataset):
                one_data = [f"{i}", "custom", data["id"], data["question"], data["question"], "", "gold"]
                for candidate in data["paragraphs"]:
                    one_data.append(contexts[candidate])
                if(split != "test"):
                    one_data.append(data["paragraphs"].index(data["relevant"]))
                writer.writerow(one_data)

        logging.info(f"Finish Process {split}")

    



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
        default="./cache/"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)

