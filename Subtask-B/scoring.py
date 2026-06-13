"""
Evaluation Script for Token-Level Classification Tasks (CoNLL format)
Compatible with Codabench

This script calculates the Macro F1 score and writes it to a scores.json 
file using the exact column keys defined in the Codabench leaderboard UI.
"""

import os
import json
from sklearn.metrics import classification_report, f1_score


def read_conll_sentences(path):
    """Read a CoNLL-style file and return sentence-level language tags."""
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
            tag = parts[1] if len(parts) >= 2 else ""
            curr_tags.append(tag)

    if curr_tags:
        sentences.append(curr_tags)

    return sentences


def evaluate_macro_f1(pred_file, true_file):
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
    
    print(f"\nPer-class metrics for current file:")
    print(classification_report(y_true, y_pred, digits=4))

    return macro_f1


def main():
    # Codabench standardizes inputs via command line arguments
    # sys.argv[1] = Input directory (contains 'res' for predictions, 'ref' for truth)
    # sys.argv[2] = Output directory (where scores.json goes)
    input_dir = '/app/input/'
    output_dir = '/app/output/'

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    # Mapping your Leaderboard "Column Keys" to the expected filenames
    # Update the values here if your files are named differently!
    TASKS = {
        'guj': 'guj.conll',  # Score for Macro F1 ENG-HIN-GUJ
        'ben': 'ben.conll'   # Score for Macro F1 ENG-HI-BEN
    }

    scores = {}

    for key, filename in TASKS.items():
        pred_file = os.path.join(submit_dir, filename)
        true_file = os.path.join(truth_dir, filename)

        if os.path.exists(pred_file) and os.path.exists(true_file):
            print(f"Evaluating {key} from {filename}...")
            try:
                score = evaluate_macro_f1(pred_file, true_file)
                scores[key] = score
            except Exception as e:
                print(f"Error evaluating {key}: {e}")
                # Assign 0.0 or handle the crash as needed for bad submissions
                scores[key] = 0.0 
        else:
            print(f"Warning: Could not find files for {key} ({filename}).")
            # If files are missing, you can either omit the key or assign 0.
            # Assigning 0 is safer so the leaderboard still populates a result.
            scores[key] = 0.0

    # Write the results to scores.json for Codabench to parse
    score_file = os.path.join(output_dir, 'scores.json')
    with open(score_file, 'w') as f:
        json.dump(scores, f)
    
    print(f"\nFinal scores written to Codabench: {scores}")


if __name__ == "__main__":
    # Codabench will invoke this script automatically
    main()