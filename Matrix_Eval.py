
import json
import sys
from collections import Counter


"""
TriMixGen-Indic Subtask A Evaluator

This script evaluates code-mixed language tag sequences using:

    • CMI (Code-Mixing Index)
    • M-Index (Multilingual Index)
    • I-Index (Integration Index)

Input:
    Two JSONL files:
        1. EN-HI-BN (HBE)
        2. EN-HI-GU (HGE)

Example record:
    {"index": 1, "EN-HI-GU": "ENG GUJ HIN ENG"}

Output:
    • Per-language-pair metric averages
    • HBE Score
    • HGE Score
    • Final Raw Score

Final Score:
    Raw Score = (HBE Score + HGE Score) / 2

Usage:
    python evaluate.py hbe.jsonl hge.jsonl
"""

# ─────────────────────────────────────────────
# METRIC 1 — CMI (Code-Mixing Index)
# Gamback & Sikdar 2016
# CMI = 100 * (1 - max_lang_tokens / non_uni_tokens)
# ─────────────────────────────────────────────
def compute_cmi(tags):
    non_uni = [t for t in tags if t != "UNI"]

    if not non_uni:
        return 0.0

    freq = Counter(non_uni)
    max_freq = max(freq.values())

    return 100.0 * (1 - max_freq / len(non_uni))


# ─────────────────────────────────────────────
# METRIC 2 — M-Index (Multilingual Index)
# Barnett et al. 2000 — language entropy based
# M = (1 - sum(p_i^2)) / (1 - 1/k)   where k = number of languages used
# ─────────────────────────────────────────────
def compute_m_index(tags):
    non_uni = [t for t in tags if t != "UNI"]

    if not non_uni:
        return 0.0

    freq = Counter(non_uni)
    k = len(freq)

    if k == 1:
        return 0.0

    n = len(non_uni)

    probs = [count / n for count in freq.values()]

    numerator = 1 - sum(p ** 2 for p in probs)
    denominator = 1 - (1 / k)

    return numerator / denominator if denominator else 0.0


# ─────────────────────────────────────────────
# METRIC 3 — I-Index (Integration Index)
# Guzmán et al. 2017
# I = number of language switch points / (n - 1)
# ─────────────────────────────────────────────
def compute_i_index(tags):
    content = [t for t in tags if t != "UNI"]

    if len(content) < 2:
        return 0.0

    switches = sum(
        1
        for i in range(1, len(content))
        if content[i] != content[i - 1]
    )

    return switches / (len(content) - 1)


# ─────────────────────────────────────────────
# Evaluate One Sentence
# ─────────────────────────────────────────────
def evaluate_row(index, tags):
    cmi = compute_cmi(tags)
    m_index = compute_m_index(tags)
    i_index = compute_i_index(tags)

    avg_score = (
        (cmi / 100.0)
        + m_index
        + i_index
    ) / 3

    return {
        "index": index,
        "cmi": round(cmi, 4),
        "m_index": round(m_index, 4),
        "i_index": round(i_index, 4),
        "avg_score": round(avg_score, 4),
    }


# ─────────────────────────────────────────────
# Process File
# ─────────────────────────────────────────────
def process_file(filepath):
    results = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                index = record["index"]

                tags_str = (
                    record.get("tags")
                    or record.get("lid_tags")
                    or record.get("language_tags")
                )

                if not tags_str:
                    continue

                tags = tags_str.strip().split()

                results.append(
                    evaluate_row(index, tags)
                )

            except Exception as e:
                print(f"Error: {e}")

    return results

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def summarize(name, results):

    if not results:
        print(f"\n{name}: No valid data")
        return 0.0

    n = len(results)

    avg_cmi = sum(r["cmi"] for r in results) / n
    avg_m   = sum(r["m_index"] for r in results) / n
    avg_i   = sum(r["i_index"] for r in results) / n

    score   = sum(r["avg_score"] for r in results) / n

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Sentences      : {n}")
    print(f"CMI Avg        : {avg_cmi:.4f}")
    print(f"M-Index Avg    : {avg_m:.4f}")
    print(f"I-Index Avg    : {avg_i:.4f}")
    print(f"Score          : {score:.4f}")

    return score


def main():

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python evaluate.py results_hbe.jsonl results_hge.jsonl"
        )
        sys.exit(1)

    hbe_file = sys.argv[1]
    hge_file = sys.argv[2]

    hbe_results = process_file(hbe_file)
    hge_results = process_file(hge_file)

    hbe_score = summarize("HBE (Hindi-Bengali-English)", hbe_results)
    hge_score = summarize("HGE (Hindi-Gujarati-English)", hge_results)

    raw_score = (hbe_score + hge_score) / 2

    print("\n")
    print("=" * 60)
    print("FINAL SUBTASK A SCORE")
    print("=" * 60)
    print(f"Raw Score : {raw_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
