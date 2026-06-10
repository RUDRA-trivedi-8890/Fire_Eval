"""
Evaluation Script for Token-Level Classification Tasks (CoNLL format)
=====================================================================

This script calculates the Macro F1 score and prints a detailed classification
report for token-level predictions (e.g., POS tagging, Named Entity Recognition)
provided in CoNLL format.

File Format Requirements (CoNLL):
--------------------------------
- Each file must be in CoNLL-style plain text where each non-empty line
    contains at least two whitespace-separated columns: a token and its tag.
    The tag is expected to be the last column on the line. Sentences are
    separated by blank lines.
- The ground-truth file (--true) contains the reference tags.
- The prediction file (--pred) contains predicted tags for the same token
    segmentation (one tag per token), in the same sentence order.

Important Features & Constraints:
---------------------------------
- Sentence alignment: The script groups tags into sentences (blocks separated
    by blank lines) and assigns incremental sentence IDs starting from 0. The
    prediction and ground-truth files are aligned by these sentence IDs.
- Token-Count Verification: For every matched sentence ID, the number of tokens
    (i.e., tags) in the ground truth MUST exactly match the number in the
    prediction. A mismatch raises a ValueError and halts.
- Missing/Null Data Handling: Empty tags are treated as empty strings.
- Optimized Processing: Uses pandas vectorized string actions to flatten lists
    efficiently for metric computation.
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_macro_f1(pred_file, true_file):
    # Read CoNLL files into DataFrames with sentence IDs and space-joined tags
    pred_df = read_conll_to_df(pred_file, col_name="pred")
    true_df = read_conll_to_df(true_file, col_name="label")

    pred_col = "pred"

    # Merge on ID
    merged = true_df.merge(pred_df[["ID", pred_col]], on="ID", how="inner")

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
        "--pred", required=True, help="Prediction CoNLL file"
    )
    parser.add_argument(
        "--true", required=True, help="Ground truth CoNLL file"
    )

    args = parser.parse_args()
    evaluate_macro_f1(pred_file=args.pred, true_file=args.true)


def read_conll_to_df(path, col_name="label"):
    """Read a CoNLL-style file and return a DataFrame with columns `ID` and
    `col_name` where each row corresponds to one sentence and the tags are a
    space-joined string of the sentence tags.

    The function expects the tag to be the last whitespace-separated column
    on each non-empty line. Sentences are separated by blank lines.
    """
    sentences = []
    curr_tags = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if curr_tags:
                    sentences.append(" ".join(curr_tags))
                    curr_tags = []
                continue
            parts = line.split()
            # Take last column as tag; if line malformed, treat tag as empty
            tag = parts[-1] if len(parts) >= 1 else ""
            curr_tags.append(tag)
        # append last sentence if file doesn't end with a blank line
        if curr_tags:
            sentences.append(" ".join(curr_tags))

    df = pd.DataFrame({
        "ID": list(range(len(sentences))),
        col_name: sentences,
    })
    return df