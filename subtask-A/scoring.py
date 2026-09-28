"""Evaluation script for the FIRE-2026 Shared Task on Code-Mixed Text Generation (Subtask A)."""
# How to run the code
# python3 evaluate_subtask_A.py
import os
import json
import math
import re
import sys
import subprocess
from collections import Counter
import gc

required_packages = ["transformers", "torch", "numpy"]

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def compute_cmi(tags):
    """Compute Code-Mixing Index (CMI) for a sentence."""
    non_uni = [t for t in tags if t != "UNI"]
    if not non_uni:
        return 0.0
    
    freq = Counter(non_uni)
    max_freq = max(freq.values())
    return 100.0 * (1 - max_freq / len(non_uni))


def compute_m_index(tags):
    """Compute Multilingual Index (M-Index) for a sentence."""
    non_uni = [t for t in tags if t != "UNI"]
    if not non_uni:
        return 0.0
    
    freq = Counter(non_uni)
    k = len(freq)
    if k == 1:
        return 0.0
    k = 3
    n = len(non_uni)
    probs = [count / n for count in freq.values()]
    numerator = 1 - sum(p ** 2 for p in probs)
    denominator = 1 - (1 / k)
    
    return numerator / denominator if denominator else 0.0


def compute_i_index(tags):
    """Compute Integration Index (I-Index) for a sentence."""
    content = [t for t in tags if t != "UNI"]
    if len(content) < 2:
        return 0.0
    
    switches = sum(1 for i in range(1, len(content)) if content[i] != content[i-1])
    return switches / (len(content) - 1)


def compute_fluency(sentence, model, tokenizer, device):
    """Compute fluency score of a sentence using MLM pseudo-log-likelihood."""
    if not sentence.strip():
        return 0.0
    
    try:
        tokenized = tokenizer(sentence, return_tensors='pt')
        input_ids = tokenized['input_ids'][0].to(device)
        word_ids = tokenized.word_ids(batch_index=0)
        
        # We evaluate each sub-token idx that has a valid word_id
        eval_token_indices = [idx for idx, w in enumerate(word_ids) if w is not None]
        if not eval_token_indices:
            return 0.0
            
        # Process in batches to avoid OOM
        fluency_batch_size = 2
        log_sum = 0.0
        
        for start_idx in range(0, len(eval_token_indices), fluency_batch_size):
            end_idx = min(start_idx + fluency_batch_size, len(eval_token_indices))
            batch_indices = eval_token_indices[start_idx:end_idx]
            
            batch_inputs = []
            for idx in batch_indices:
                w_idx = word_ids[idx]
                # Find all tokens belonging to the same word
                t_indices = [k for k, w in enumerate(word_ids) if w == w_idx]
                
                # Position of current token within the word's sub-tokens
                j = t_indices.index(idx)
                
                # Mask current and all subsequent tokens of this word (PLL-word-l2r)
                masked_ids = input_ids.clone()
                for mask_idx in t_indices[j:]:
                    masked_ids[mask_idx] = tokenizer.mask_token_id
                batch_inputs.append(masked_ids)
                
            batch_inputs = torch.stack(batch_inputs).to(device)
            
            with torch.no_grad():
                logits = model(batch_inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            for i, idx in enumerate(batch_indices):
                correct_token = input_ids[idx].item()
                prob = probs[i, idx, correct_token].item()
                log_sum += math.log(max(prob, 1e-9))
            
            # Clear memory after each batch
            del batch_inputs, logits, probs
            torch.cuda.empty_cache()
            
        val = math.exp(log_sum / len(eval_token_indices))
        return math.pow(val, 0.25)
        
    except Exception as e:
        raise RuntimeError(f"Fluency model prediction failed: {e}") from e


def evaluate_sentence(index, tags, text, model, tokenizer, device, similarity_score):
    """Evaluate all metrics for a single sentence."""
    cmi = compute_cmi(tags)
    m_index = compute_m_index(tags)
    i_index = compute_i_index(tags)
    fluency = compute_fluency(text, model, tokenizer, device)
    
    cmi_norm = cmi / 100.0
    mixing_avg = (cmi_norm + m_index + i_index) / 3.0
    avg_score = (mixing_avg + fluency + similarity_score) / 3.0
    
    return {
        "index": index,
        "cmi": round(cmi, 4),
        "cmi_norm": round(cmi_norm, 4),
        "m_index": round(m_index, 4),
        "i_index": round(i_index, 4),
        "fluency": round(fluency, 4),
        "similarity": round(similarity_score, 4),
        "avg_score": round(avg_score, 4)
    }


def calculate_average(results, name):
    """Calculate and print average metrics across the dataset."""
    if not results:
        print(f"\n{name}: No data")
        return None
    
    n = len(results)
    metrics = {
        'cmi': sum(r['cmi'] for r in results) / n,
        'cmi_norm': sum(r['cmi_norm'] for r in results) / n,
        'm_index': sum(r['m_index'] for r in results) / n,
        'i_index': sum(r['i_index'] for r in results) / n,
        'fluency': sum(r['fluency'] for r in results) / n,
        'similarity': sum(r['similarity'] for r in results) / n,
        'avg_score': sum(r['avg_score'] for r in results) / n
    }
    
    print("\n" + "="*60)
    print(f"{name}")
    print("="*60)
    print(f"Sentences       : {n}")
    print(f"CMI (raw)       : {metrics['cmi']:.4f}")
    print(f"CMI (norm)      : {metrics['cmi_norm']:.4f}")
    print(f"M-Index         : {metrics['m_index']:.4f}")
    print(f"I-Index         : {metrics['i_index']:.4f}")
    print(f"Fluency         : {metrics['fluency']:.4f}")
    print(f"Similarity      : {metrics['similarity']:.4f}")
    print(f"AVG SCORE       : {metrics['avg_score']:.4f}")
    print("="*60)
    
    return metrics['avg_score']


class Scoring:
    """Main Scoring class for the evaluation program."""
    def __init__(self, answer_dir, prediction_dir, output_dir):
        self.answer_dir = answer_dir
        self.prediction_dir = prediction_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load LID model
        model_dir = "/app/data/LID-model"
        if not os.path.exists(model_dir):
            model_dir = os.path.join(self.answer_dir, "LID-model")

        print("Loading LID tokenizer...")
        self.lid_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Loading LID model...")
        self.lid_model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            use_safetensors=True,
            from_tf=False,
            from_flax=False
        ).to(self.device)
        self.lid_model.eval()
        self.id2label = self.lid_model.config.id2label
        self.lid_type = "token"
        print("LID model loaded successfully\n")
        
        # Load fluency model
        fluency_model_dir = "/app/data/Fluency-model"
        if not os.path.exists(fluency_model_dir):
            fluency_model_dir = os.path.join(self.answer_dir, "Fluency-model")

        print("Loading fluency model...")
        self.tokenizer = AutoTokenizer.from_pretrained(fluency_model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(fluency_model_dir).to(self.device)
        self.model.eval()
        print("Fluency model loaded\n")

        # Load sentence similarity model
        sim_model_dir = "/app/data/sentence-similarity-model"
        if not os.path.exists(sim_model_dir):
            sim_model_dir = os.path.join(self.answer_dir, "sentence-similarity-model")

        print(f"Loading sentence similarity model from {sim_model_dir}...")
        self.similarity_tokenizer = AutoTokenizer.from_pretrained(sim_model_dir, use_fast=True)
        self.similarity_model = AutoModel.from_pretrained(sim_model_dir).to(self.device)
        self.similarity_model.eval()
        print("Sentence similarity model loaded successfully\n")
    
    def evaluate(self):
        print("=" * 60)
        print("SCORING STARTED")
        print("=" * 60)
        print(f"Reference directory : {self.answer_dir}")
        print(f"Prediction directory: {self.prediction_dir}")
        print(f"Output directory    : {self.output_dir}")

        TASKS = {
            "hbe_score": "eng-hin-ben.jsonl",
            "hge_score": "eng-hin-guj.jsonl"
        }

        # Check sentence count for each task file before evaluation
        found_any = False
        for score_key, filename in TASKS.items():
            filepath = os.path.join(self.prediction_dir, filename)
            
            # Case-insensitive fallback
            if not os.path.exists(filepath):
                found_file = None
                if os.path.isdir(self.prediction_dir):
                    for f in os.listdir(self.prediction_dir):
                        if f.lower() == filename.lower():
                            found_file = f
                            break
                if found_file:
                    filepath = os.path.join(self.prediction_dir, found_file)
                else:
                    continue

            # At this point we have a filepath that exists
            found_any = True

            line_count = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        line_count += 1
            
            if line_count != 500:
                raise ValueError(
                    f"File {filename} has {line_count} sentences, which is not the required number of 500 sentences."
                )

        # If none of the required prediction files are present, raise an error
        if not found_any:
            raise FileNotFoundError(
                "Neither eng-hin-ben nor eng-hin-guj prediction files were found in the prediction directory."
            )

        scores = {}

        for score_key, filename in TASKS.items():
            filepath = os.path.join(self.prediction_dir, filename)

            if not os.path.exists(filepath):
                # Fallback to case-insensitive match
                found_file = None
                if os.path.isdir(self.prediction_dir):
                    for f in os.listdir(self.prediction_dir):
                        if f.lower() == filename.lower():
                            found_file = f
                            break
                if found_file:
                    filepath = os.path.join(self.prediction_dir, found_file)
                else:
                    print(f"Missing prediction file: {filename}")
                    scores[score_key] = -1.0
                    continue

            results = self.process_file(filepath)

            if not results:
                print(f"No valid records in {filename}")
                scores[score_key] = -1.0
                continue

            dataset_score = calculate_average(results, score_key)
            scores[score_key] = round(dataset_score, 4)

        print("\nFinal Scores")
        print(scores)
        self.write_results(scores)

    def process_file(self, filepath):
        """Process prediction JSONL file and return list of sentence metrics."""
        results = []
        file_name = os.path.basename(filepath)
        print(f"Processing: {file_name}")

        if "ben" in file_name.lower():
            dev_filename = "dev_eng_hin_ben.jsonl"
            third_lang_key = "bn"
        elif "guj" in file_name.lower():
            dev_filename = "dev_eng_hin_guj.jsonl"
            third_lang_key = "gu"
        else:
            raise ValueError(f"Unknown language in prediction filename: {file_name}")

        dev_filepath = os.path.join(self.answer_dir, "dev-data", dev_filename)
        if not os.path.exists(dev_filepath):
            dev_filepath = f"/app/data/dev-data/{dev_filename}"
        if not os.path.exists(dev_filepath):
            dev_filepath = f"/app/input/ref/dev-data/{dev_filename}"

        dev_records = []
        if os.path.exists(dev_filepath):
            print(f"Loading dev-data references from {dev_filepath}...")
            with open(dev_filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            dev_records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in dev-data at line {line_num}: {e}")
        else:
            raise FileNotFoundError(f"dev-data file not found at {dev_filepath}")

        pred_records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "output" not in record and "generated" not in record:
                            raise ValueError(
                                f"Missing 'output' or 'generated' field in prediction file {file_name} at line {line_num}"
                            )
                        pred_records.append(record)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON format in prediction file {file_name} at line {line_num}: {e}")

        pred_sentences = []
        for rec in pred_records:
            sentence = rec.get("output", rec.get("generated", ""))
            pred_sentences.append(sentence if sentence is not None else "")

        if dev_records:
            similarity_scores = self.compute_similarity_scores(pred_sentences, dev_records, third_lang_key)
        else:
            similarity_scores = [0.0] * len(pred_sentences)

        for line_num, record in enumerate(pred_records, 1):
            try:
                index = record.get("index", line_num - 1)
                text = record.get("output", record.get("generated"))
                if text is None:
                    raise ValueError(f"'output' or 'generated' field is None at line {line_num}")
                text = str(text)

                tags = self.predict_tags(text) if text.strip() else []
                sim_score = similarity_scores[line_num - 1] if (line_num - 1) < len(similarity_scores) else 0.0

                result = evaluate_sentence(
                    index,
                    tags,
                    text,
                    self.model,
                    self.tokenizer,
                    self.device,
                    sim_score,
                )
                results.append(result)
                
                # Clear memory periodically
                if line_num % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                raise RuntimeError(f"Failed to process line {line_num} in {file_name}: {e}") from e

        print(f"Processed {len(results)} sentences")
        return results

    def compute_similarity_scores(self, pred_sentences, dev_records, third_lang_key):
        """Compute sentence similarity scores for a batch of predictions."""
        self.similarity_model.eval()

        def encode_batch(sentences, batch_size=16):
            all_embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                batch = [s if s.strip() else " " for s in batch]
                inputs = self.similarity_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.similarity_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                
                # Clear memory after each batch
                del inputs, outputs, embeddings
                torch.cuda.empty_cache()
                
            return torch.cat(all_embeddings, dim=0)

        n = min(len(pred_sentences), len(dev_records))
        if n == 0:
            return [0.0] * len(pred_sentences)

        # Process in chunks to avoid OOM
        chunk_size = 50
        all_scores = []
        
        for start_idx in range(0, n, chunk_size):
            end_idx = min(start_idx + chunk_size, n)
            
            ref_en = [dev_records[i]["en"] for i in range(start_idx, end_idx)]
            ref_hi = [dev_records[i]["hi"] for i in range(start_idx, end_idx)]
            ref_lang3 = [dev_records[i][third_lang_key] for i in range(start_idx, end_idx)]
            pred_chunk = pred_sentences[start_idx:end_idx]

            pred_embs = encode_batch(pred_chunk)
            en_embs = encode_batch(ref_en)
            hi_embs = encode_batch(ref_hi)
            lang3_embs = encode_batch(ref_lang3)

            sim_en = torch.sum(pred_embs * en_embs, dim=1).numpy()
            sim_hi = torch.sum(pred_embs * hi_embs, dim=1).numpy()
            sim_lang3 = torch.sum(pred_embs * lang3_embs, dim=1).numpy()

            avg_similarities = (sim_en + sim_hi + sim_lang3) / 3.0
            all_scores.extend(avg_similarities.tolist())
            
            # Clear memory after each chunk
            del pred_embs, en_embs, hi_embs, lang3_embs, sim_en, sim_hi, sim_lang3
            torch.cuda.empty_cache()
            gc.collect()
        
        scores = all_scores
        if len(pred_sentences) > n:
            scores.extend([0.0] * (len(pred_sentences) - n))

        return scores
    
    def predict_tags(self, text):
        """Predict language tag sequence for a sentence using LID model."""
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        if not tokens:
            return []

        try:
            tokenized_inputs = self.lid_tokenizer(
                tokens,
                truncation=True,
                is_split_into_words=True,
                max_length=256,
                return_tensors="pt",
                padding=False
            )

            input_ids = tokenized_inputs["input_ids"].to(self.device)
            attention_mask = tokenized_inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.lid_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            predictions = outputs.logits.cpu().numpy()
            predictions = np.argmax(predictions, axis=2)

            word_ids = tokenized_inputs.word_ids(batch_index=0)

            predicted_tags = []
            current_word_idx = None

            for pred, word_idx in zip(predictions[0], word_ids):
                if word_idx is None:
                    continue

                if word_idx != current_word_idx:
                    predicted_tags.append(self.id2label[int(pred)])
                    current_word_idx = word_idx

            if len(predicted_tags) < len(tokens):
                predicted_tags.extend(["UNI"] * (len(tokens) - len(predicted_tags)))
            elif len(predicted_tags) > len(tokens):
                predicted_tags = predicted_tags[:len(tokens)]

            return predicted_tags

        except Exception as e:
            raise RuntimeError(f"LID model prediction failed: {e}") from e

    def write_results(self, scores):
        """Write final scores to JSON file."""
        score_file = os.path.join(self.output_dir, "scores.json")
        with open(score_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)

        print(f"\nScores written to:")
        print(score_file)
        print(json.dumps(scores, indent=2))


def main():
    """Pass arguments to the Scoring class and initiate evaluation."""
    input_dir = "Input Directory Path"
    output_dir = "Output Directory Path"

    answer_dir = os.path.join(input_dir, "ref") # Gold 
    prediction_dir = os.path.join(input_dir, "res")

    os.makedirs(output_dir, exist_ok=True)

    scoring = Scoring(
        answer_dir=answer_dir,
        prediction_dir=prediction_dir,
        output_dir=output_dir
    )
    scoring.evaluate()


if __name__ == "__main__":
    main()