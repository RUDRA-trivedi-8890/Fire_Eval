"""
Evaluation Script for Token-Level Classification Tasks (CoNLL format)
=====================================================================

This script calculates the Macro F1 score and prints a detailed classification
report for token-level predictions (e.g., language identification at token level).

File Format Requirements:
-----------------------
- Each file must be in CoNLL-style plain text where each non-empty line
    contains two tab-separated columns: a token and its language tag.
- Sentences are separated by blank lines.
- The ground-truth file (--true) contains the reference language tags.
- The prediction file (--pred) contains predicted language tags for the same token
    segmentation (one tag per token), in the same sentence order.

Important Features & Constraints:
---------------------------------
- Sentence alignment: The script groups tags into sentences (blocks separated
    by blank lines). Sentences are aligned by their position in the files.
- Token-Count Verification: For every sentence, the number of tokens in the 
    ground truth MUST exactly match the number in the prediction. A mismatch 
    raises a ValueError and halts.
- Missing/Null Data Handling: Empty tags are treated as empty strings.
- Optimized Processing: Uses pandas vectorized string actions to flatten lists
    efficiently for metric computation.
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_macro_f1(pred_file, true_file):
    # Read CoNLL files into DataFrames with sentence tags
    pred_sentences = read_conll_sentences(pred_file)
    true_sentences = read_conll_sentences(true_file)

    # Verify sentence counts match
    if len(pred_sentences) != len(true_sentences):
        raise ValueError(
            f"Sentence count mismatch: predicted={len(pred_sentences)}, "
            f"ground-truth={len(true_sentences)}"
        )

    # Verify token counts match for each sentence
    for sent_id, (true_tags, pred_tags) in enumerate(zip(true_sentences, pred_sentences)):
        if len(true_tags) != len(pred_tags):
            raise ValueError(
                f"Token count mismatch for sentence {sent_id} "
                f"(true={len(true_tags)}, pred={len(pred_tags)})"
            )

    # Flatten all tags across all sentences
    y_true = [tag for sentence_tags in true_sentences for tag in sentence_tags]
    y_pred = [tag for sentence_tags in pred_sentences for tag in sentence_tags]

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


def read_conll_sentences(path):
    """Read a CoNLL-style file and return a list of sentences, where each 
    sentence is a list of language tags.

    The function expects each non-empty line to have format: word\tlanguage
    where the language tag is the second tab-separated column.
    Sentences are separated by blank lines.
    """
    sentences = []
    curr_tags = []
    
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if curr_tags:
                    sentences.append(curr_tags)
                    curr_tags = []
                continue
            parts = line.split("\t")
            # Extract language tag (second column)
            tag = parts[1] if len(parts) >= 2 else ""
            curr_tags.append(tag)
        # append last sentence if file doesn't end with a blank line
        if curr_tags:
            sentences.append(curr_tags)

    return sentences