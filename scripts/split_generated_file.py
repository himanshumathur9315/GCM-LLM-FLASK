import json
import random
import argparse

def split_jsonl(input_file, train_file, val_file, split_ratio=0.8):
    """
    Splits a JSONL file into training and validation sets.

    Args:
        input_file (str): The path to the input JSONL file.
        train_file (str): The path to the output training JSONL file.
        val_file (str): The path to the output validation JSONL file.
        split_ratio (float): The ratio of data to be used for training (0.0 to 1.0).
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Shuffle the lines to ensure random distribution
        random.shuffle(lines)

        # Calculate the split index
        split_index = int(len(lines) * split_ratio)

        # Split the data
        train_lines = lines[:split_index]
        val_lines = lines[split_index:]

        # Write the training set
        with open(train_file, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line)
        print(f"Successfully created training set with {len(train_lines)} records at: {train_file}")

        # Write the validation set
        with open(val_file, 'w', encoding='utf-8') as f:
            for line in val_lines:
                f.write(line)
        print(f"Successfully created validation set with {len(val_lines)} records at: {val_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL file into training and validation sets.")
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("--train_file", default="training_set.jsonl", help="Path for the output training file.")
    parser.add_argument("--val_file", default="validation_set.jsonl", help="Path for the output validation file.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Split ratio for the training set (e.g., 0.8 for 80%).")

    args = parser.parse_args()

    # To run this script from your terminal:
    # python split_dataset.py generated_sft_data_simulator.jsonl --train_file training.jsonl --val_file validation.jsonl --ratio 0.8
    split_jsonl(args.input_file, args.train_file, args.val_file, args.ratio)
