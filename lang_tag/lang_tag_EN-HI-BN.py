"""
Word-level language identification for Romanized code-mixed text
using Ollama Chat API.

Improvements:
- Uses /api/chat
- Deterministic decoding
- Numbered token input
- Explicit token count enforcement
- Output validation
- Automatic retries
"""

import json
import sys
from pathlib import Path
import requests

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma4:31b-cloud"

SYSTEM_PROMPT = """
You are an expert word-level language identification system.

Available labels:

HIN = Hindi
BN = Bengali
ENG = English
UNI = Universal / Named Entity / Number / Unknown

Rules:

1. Every token MUST receive exactly one label.
2. Number of labels MUST equal number of tokens.
3. Preserve token order.
4. Output ONLY space-separated labels.
5. No explanations.
6. No markdown.
7. No punctuation.
8. No extra text.

Example:

Tokens:
1. Aaj
2. mujhe
3. der
4. ho
5. gayi
6. kyunki
7. traffic
8. vahu
9. hatu

Labels:
HIN HIN HIN HIN HIN HIN ENG BN BN
"""


def tokenize_sentence(text):
    return text.split()


def build_user_prompt(text):
    tokens = tokenize_sentence(text)

    token_lines = "\n".join(
        f"{i+1}. {token}"
        for i, token in enumerate(tokens)
    )

    return f"""
There are exactly {len(tokens)} tokens below.

You MUST return exactly {len(tokens)} labels.

Available labels:
HIN
BN
ENG
UNI

Tokens:
{token_lines}

Return only space-separated labels.

Labels:
""".strip()


def validate_output(text, labels):
    token_count = len(tokenize_sentence(text))
    label_count = len(labels.split())

    return token_count == label_count


def call_ollama_api(text, max_retries=5):

    expected = len(tokenize_sentence(text))

    for attempt in range(max_retries):

        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(text)
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "num_predict": expected * 3
                }
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        labels = result["message"]["content"].strip()

        labels = " ".join(labels.split())

        if validate_output(text, labels):
            return labels

        print(
            f"[Retry {attempt+1}/{max_retries}] "
            f"Expected {expected} labels, "
            f"got {len(labels.split())}"
        )

    print("WARNING: Returning best attempt despite mismatch.")
    return labels


def process_jsonl(input_file, output_dir):

    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir /
        f"{input_path.stem}__EN-HI-BN.jsonl"
    )

    processed = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid JSON at line {line_num}"
                )
                continue

            index = record.get("index")

            text = record.get("EN-HI-BN")

            if not text:
                continue

            print(
                f"[Line {line_num}] "
                f"Processing index={index}"
            )

            labels = call_ollama_api(text)

            token_count = len(tokenize_sentence(text))
            label_count = len(labels.split())

            print(
                f"Tokens={token_count} "
                f"Labels={label_count}"
            )

            print(f"Sentence: {text}")
            print(f"Output:   {labels}")
            print()

            output_record = {
                "index": index,
                "EN-HI-BN": labels
            }

            outfile.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False
                ) + "\n"
            )

            outfile.flush()

            processed += 1

    print("\n" + "=" * 60)
    print(f"Finished.")
    print(f"Processed: {processed}")
    print(f"Output: {output_file}")


if __name__ == "__main__":

    INPUT_FILE = "en-hi-bn_code_mixed_data.jsonl"
    OUTPUT_DIR = "output_tags"

    process_jsonl(
        INPUT_FILE,
        OUTPUT_DIR
    )
