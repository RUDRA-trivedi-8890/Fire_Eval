"""
TriMixGen-Indic Subtask A Evaluator

================================================================================
1. DATA INPUT & PARSING LOGIC
================================================================================
The script streams .jsonl files line-by-line and applies flexible fallback logic 
to support variation in field names across your datasets:

• Language Tags Key: Looks for 'tags', 'lid_tags', or 'language_tags'.
• Raw Text Key (Fluency): Looks for 'text', 'sentence', or the custom language-pair 
  fallback names ('EN-HI-BN' / 'EN-HI-GU').

Supported JSON line variations:
  Format A: {"index": 1, "tags": "ENG HIN", "text": "Oi Bollywood..."}
  Format B: {"index": 1, "language_tags": "ENG HIN", "EN-HI-BN": "Oi Bollywood..."}

================================================================================
2. METRIC FORMULAS & CALCULATION LOGIC
================================================================================
All metrics below run on a matching 0.0 to 1.0 positive scale (Higher = Better):

• Normalized CMI:   CMI_norm = (100 * (1 - (max_lang_tokens / non_uni_tokens))) / 100
• M-Index:         M = (1 - sum(p_i^2)) / (1 - 1/k)
• I-Index:         I = language_switch_points / (n - 1)
• Fluency Score:   Prob = Product( P(w_i | W_{\i}) ) ^ (1/N) [Geometric Mean Prob]

================================================================================
3. DIRECT AVERAGING SYSTEM
================================================================================
• Tier 1: Per-Sentence Metrics
  Because all values are clean, positive decimals between 0.0 and 1.0, they can 
  be directly averaged and compared. 
  avg_score = (CMI_norm + M-Index + I-Index) / 3

• Tier 2: System-Level Metric Combination
  Aggregates metrics inside each file, then computes the final even split:
  Final Code-Mix Metric Score = (HBE Raw Code Score + HGE Raw Code Score) / 2
  Final Fluency Score          = (HBE Fluency Score Avg + HGE Fluency Score Avg) / 2

Usage:
    python evaluate.py hbe.jsonl hge.jsonl
"""

import json
import sys
import math
from collections import Counter
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────
# METRIC 1 — CMI (Code-Mixing Index)
# Formula: CMI = 100 * (1 - (max_lang_tokens / non_uni_tokens))
# ──────────────────────────────────────────────────────────────────────
def compute_cmi(tags):
    non_uni = [t for t in tags if t != "UNI"]
    if not non_uni:
        return 0.0

    freq = Counter(non_uni)
    max_freq = max(freq.values())
    return 100.0 * (1 - max_freq / len(non_uni))


# ──────────────────────────────────────────────────────────────────────
# METRIC 2 — M-Index (Multilingual Index)
# Formula: M = (1 - sum(p_i^2)) / (1 - 1/k)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# METRIC 3 — I-Index (Integration Index)
# Formula: I = language_switch_points / (n - 1)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# METRIC 4 — Sequence Probability Fluency Score (No Logs Output)
# Formula: Prob = Product( P(w_i | W_{\i}) ) ^ (1/N)
# ──────────────────────────────────────────────────────────────────────
def compute_sentence_probability(sentence, model, tokenizer, device):
    if not sentence.strip():
        return 0.0

    tokens = tokenizer.encode(sentence, return_tensors='pt').to(device)
    seq_len = tokens.size(1)
    
    num_eval_tokens = seq_len - 2  # Exclude <s> and </s>
    if num_eval_tokens <= 0:
        return 0.0

    batch_tokens = tokens.repeat(num_eval_tokens, 1)
    for i in range(num_eval_tokens):
        batch_tokens[i, i + 1] = tokenizer.mask_token_id
        
    with torch.no_grad():
        logits = model(batch_tokens).logits
        
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    total_geometric_log_sum = 0.0
    for i in range(num_eval_tokens):
        actual_pos = i + 1
        target_token_id = tokens[0, actual_pos]
        token_prob = probs[i, actual_pos, target_token_id].item()
        
        # Local logs used inside loop strictly to bypass float underflow errors
        total_geometric_log_sum += math.log(max(token_prob, 1e-9))
        
    final_sentence_prob = math.exp(total_geometric_log_sum / num_eval_tokens)
    return final_sentence_prob


# ──────────────────────────────────────────────────────────────────────
# Evaluate One Row
# ──────────────────────────────────────────────────────────────────────
def evaluate_row(index, tags, text, model, tokenizer, device):
    cmi = compute_cmi(tags)
    m_index = compute_m_index(tags)
    i_index = compute_i_index(tags)
    fluency_score = compute_sentence_probability(text, model, tokenizer, device)

    # Tier 1 Direct Balance (All terms map from 0.0 to 1.0)
    avg_score = ((cmi / 100.0) + m_index + i_index) / 3

    return {
        "index": index,
        "cmi": round(cmi, 4),
        "m_index": round(m_index, 4),
        "i_index": round(i_index, 4),
        "avg_score": round(avg_score, 4),
        "fluency_score": round(fluency_score, 4)
    }


# ──────────────────────────────────────────────────────────────────────
# Process File
# ──────────────────────────────────────────────────────────────────────
def process_file(filepath, model, tokenizer, device, key_identifier):
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
                    record.get("tags") or 
                    record.get("lid_tags") or 
                    record.get("language_tags")
                )
                
                text_str = (
                    record.get("text") or 
                    record.get("sentence") or 
                    record.get(key_identifier)
                )

                if not tags_str or not text_str:
                    continue

                tags = tags_str.strip().split()
                results.append(
                    evaluate_row(index, tags, text_str, model, tokenizer, device)
                )

            except Exception as e:
                print(f"Error in {filepath}: {e}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Summarize Results
# ──────────────────────────────────────────────────────────────────────
def summarize(name, results):
    if not results:
        print(f"\n{name}: No valid data found.")
        return 0.0, 0.0

    n = len(results)
    avg_cmi = sum(r["cmi"] for r in results) / n
    avg_m   = sum(r["m_index"] for r in results) / n
    avg_i   = sum(r["i_index"] for r in results) / n
    score   = sum(r["avg_score"] for r in results) / n
    avg_flu = sum(r["fluency_score"] for r in results) / n

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Sentences      : {n}")
    print(f"CMI Avg        : {avg_cmi:.4f}")
    print(f"M-Index Avg    : {avg_m:.4f}")
    print(f"I-Index Avg    : {avg_i:.4f}")
    print(f"Fluency Avg    : {avg_flu:.4f} (Direct 0-1 Probability)")
    print(f"Raw Code Score : {score:.4f}")

    return score, avg_flu


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py results_hbe.jsonl results_hge.jsonl")
        sys.exit(1)

    hbe_file = sys.argv[1]
    hge_file = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "xlm-roberta-base"
    
    print(f"Loading {model_name} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    hbe_results = process_file(hbe_file, model, tokenizer, device, "EN-HI-BN")
    hge_results = process_file(hge_file, model, tokenizer, device, "EN-HI-GU")

    hbe_score, hbe_flu = summarize("HBE (Hindi-Bengali-English)", hbe_results)
    hge_score, hge_flu = summarize("HGE (Hindi-Gujarati-English)", hge_results)

    # Tier 2 Even Weight Averages
    final_raw_score = (hbe_score + hge_score) / 2
    final_flu_score = (hbe_flu + hge_flu) / 2

    print("\n")
    print("=" * 60)
    print("FINAL COMBINED SUBTASK A SCORE")
    print("=" * 60)
    print(f"Final Code-Mix Metric Score : {final_raw_score:.4f}")
    print(f"Final Fluency Score (Prob)  : {final_flu_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
