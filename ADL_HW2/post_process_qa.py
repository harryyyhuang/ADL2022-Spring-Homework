
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

    predictions = json.loads(args.data_file.read_text())

    output_path = args.output_file 

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        for key, value in predictions.items():
            writer.writerow([key, value])

    logging.info(f"Finish PostProcess.")

    

def parser_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=Path,
        help="Directory to the dataset.",
        default="./predict/test_predictions.json"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Directory to save the processed file",
        default="./predict/test_prediction.csv"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser_args()
    main(args)