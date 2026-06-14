import os
import sys
import json
from collections import Counter

def compute_cmi(tags_list):
    non_uni = [t for t in tags_list if t != "UNI" and t != ""]
    if not non_uni: 
        return 0.0
    return 100.0 * (1 - max(Counter(non_uni).values()) / len(non_uni))

def compute_m_index(tags_list):
    non_uni = [t for t in tags_list if t != "UNI" and t != ""]
    if not non_uni: 
        return 0.0
    freq = Counter(non_uni)
    k = len(freq)
    if k == 1: 
        return 0.0
    probs = [count / len(non_uni) for count in freq.values()]
    return (1 - sum(p ** 2 for p in probs)) / (1 - (1 / k))

def compute_i_index(tags_list):
    content = [t for t in tags_list if t != "UNI" and t != ""]
    if len(content) < 2: 
        return 0.0
    switches = sum(1 for i in range(1, len(content)) if content[i] != content[i-1])
    return switches / (len(content) - 1)

def main():
    print("=== SCORING PHASE: CALCULATING COMPUTATIONAL METRICS ===")
    
    # ✅ FIXED: Correct path to bridge file
    input_dir = '/app/output/res'  # Where ingestion writes
    output_dir = '/app/output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    scores = {"guj": 0.0, "ben": 0.0}
    
    bridge_file_path = os.path.join(input_dir, "ingested_payload.json")
    print(f"🔍 Looking for bridge file at: {bridge_file_path}")
    
    if os.path.exists(bridge_file_path):
        print(f"✅ Found bridge file!")
        
        try:
            with open(bridge_file_path, "r", encoding="utf-8") as f:
                payload_data = json.load(f)
            
            for key in ["guj", "ben"]:
                items = payload_data.get(key, [])
                print(f"\nProcessing {key.upper()}: {len(items)} samples")
                
                if not items:
                    continue
                
                track_scores = []
                for sample in items:
                    tags_str = sample.get("predicted_tags", "")
                    tags_list = tags_str.split()
                    
                    cmi_raw = compute_cmi(tags_list)
                    cmi_norm = cmi_raw / 100.0
                    m_idx = compute_m_index(tags_list)
                    i_idx = compute_i_index(tags_list)
                    
                    sentence_score = (cmi_norm + m_idx + i_idx) / 3
                    track_scores.append(sentence_score)
                
                if track_scores:
                    scores[key] = round(sum(track_scores) / len(track_scores), 4)
                    print(f"  ✅ Score for {key.upper()}: {scores[key]}")
            
        except Exception as e:
            print(f"❌ Error processing scores: {e}")
    else:
        print(f"❌ Bridge file not found!")
    
    # Save scores
    score_file_path = os.path.join(output_dir, "scores.json")
    with open(score_file_path, "w", encoding="utf-8") as out_f:
        json.dump(scores, out_f, indent=4)
    
    print(f"\n✅ Scores saved to: {score_file_path}")
    print(f"🚀 Evaluation Complete. Final scores: {scores}")

if __name__ == "__main__":
    main()