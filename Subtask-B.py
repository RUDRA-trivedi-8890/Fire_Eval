"""
Evaluation Script for Token-Level Classification Tasks
======================================================

This script calculates the Macro F1 score and provides a detailed classification
report for token-level predictions (e.g., POS tagging, Named Entity Recognition)
stored in CSV format.

File Format Requirements:
-------------------------
1. Ground Truth File (--true):
   - Must be a CSV file.
   - Must contain at least two columns: 'ID' and 'label'.
   - The 'label' column should contain space-separated tokens/tags for each sequence.
     Example row: 
     ID,label
     123,"B-PER I-PER O O B-LOC"

2. Prediction File (--pred):
   - Must be a CSV file.
   - Must contain an 'ID' column to match against the ground truth.
   - Must contain exactly *one* other column containing the predicted tags.
     The script will automatically detect this column name.
   - The prediction column must contain space-separated tokens/tags.
     Example row:
     ID,model_predictions
     123,"B-PER O O O B-LOC"

Important Features & Constraints:
---------------------------------
- Inner Join Alignment: The script merges both files on the 'ID' column. If the 
  prediction file is missing IDs present in the ground truth, a warning is printed.
- Token-Count Verification: For every matched ID, the number of space-separated 
  tokens in the ground truth MUST exactly match the number of tokens in the 
  prediction. If a mismatch is found (e.g., row 123 has 5 true tags but 4 predicted 
  tags), the script will raise a ValueError and halt.
- Missing/Null Data Handling: Empty fields or NaNs are automatically treated as 
  empty strings to prevent common pandas string-parsing bugs (like reading empty 
  rows as the string "nan").
- Optimized Processing: Uses vectorized pandas string actions to flatten arrays 
  efficiently, offering high performance even on large datasets.
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_macro_f1(pred_file, true_file):
    # Read files
    pred_df = pd.read_csv(pred_file)
    true_df = pd.read_csv(true_file)

    # Validate required columns
    if "ID" not in pred_df.columns:
        raise ValueError("Prediction file must contain an 'ID' column.")

    if "ID" not in true_df.columns or "label" not in true_df.columns:
        raise ValueError(
            "Ground truth file must contain 'ID' and 'label' columns."
        )

    # Find prediction column automatically
    pred_cols = [c for c in pred_df.columns if c != "ID"]
    if len(pred_cols) != 1:
        raise ValueError(
            f"Prediction file should contain exactly one prediction column besides ID. Found: {pred_cols}"
        )
    pred_col = pred_cols[0]

    # Merge on ID
    merged = true_df.merge(
        pred_df[["ID", pred_col]], on="ID", how="inner"
    )

    if len(merged) != len(true_df):
        print(
            f"Warning: matched {len(merged)} of {len(true_df)} ground-truth rows."
        )

    # Fill NaN values with empty strings to prevent "nan" token bugs
    merged["label"] = merged["label"].fillna("").astype(str)
    merged[pred_col] = merged[pred_col].fillna("").astype(str)

    # Vectorized splitting
    true_lists = merged["label"].str.split()
    pred_lists = merged[pred_col].str.split()

    # Fast verification of token counts per row
    lengths_match = true_lists.str.len() == pred_lists.str.len()
    if not lengths_match.all():
        # Find the first offending row to show in the error
        failed_row = merged[~lengths_match].iloc[0]
        t_len = len(true_lists.loc[failed_row.name])
        p_len = len(pred_lists.loc[failed_row.name])
        raise ValueError(
            f"Token count mismatch for ID={failed_row['ID']} "
            f"(true={t_len}, pred={p_len})"
        )

    # Flatten the series of lists into single lists
    y_true = [token for sublist in true_lists for token in sublist]
    y_pred = [token for sublist in pred_lists for token in sublist]

    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nMacro F1: {macro_f1:.6f}\n")
    print("Per-class metrics:")
    print(classification_report(y_true, y_pred, digits=4))

    return macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred", required=True, help="Prediction CSV"
    )
    parser.add_argument(
        "--true", required=True, help="Ground truth CSV"
    )

    args = parser.parse_args()
    evaluate_macro_f1(pred_file=args.pred, true_file=args.true)