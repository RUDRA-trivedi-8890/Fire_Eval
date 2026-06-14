import os
import sys
import json
import subprocess

# --- AUTOMATIC PIP DEPENDENCY RESOLVER ---
try:
    import requests
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests

def call_gemini_api(text, api_key):
    """Call Gemini 2.5 Flash API for language tagging"""
    
    # ✅ FIXED: Use v1beta (not v1) for Gemini 2.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    prompt = f"""You are a linguistic token tagger. Analyze the following sentence and tag each individual word.
Use ENG for English, HIN for Hindi, GUJ for Gujarati, BEN for Bengali, and UNI for punctuation or symbols.
Return ONLY the tags separated by spaces, matching the exact order and count of words. Do not give any markdown or explanation.

Sentence: {text}"""
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 500
        }
    }
    
    try:
        print(f"  📡 Calling Gemini 2.5 Flash API...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                tags = result['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"  ✅ API Success: {tags[:100]}...")
                return tags
            else:
                print(f"  ⚠️ Unexpected response format")
                return None
        else:
            print(f"  ❌ API Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"  ⚠️ API Exception: {e}")
        return None

def find_file_in_codabench_paths(target_filename):
    """Search all possible Codabench mount points for target files"""
    # ✅ FIXED: Added /app/ingested_program where your files actually are
    search_paths = [
        '/app/ingested_program',    # Where your files are located
        '/app/program',              
        '/app/data',                 
        '/app/input',                
        '/app/input_data',           
        '/app',                      
    ]
    
    for directory in search_paths:
        if not os.path.exists(directory):
            continue
        
        for root, dirs, files in os.walk(directory):
            if any(skip in root for skip in ['.local', '__pycache__', '.cache']):
                continue
            if target_filename in files:
                full_path = os.path.join(root, target_filename)
                print(f"  ✅ Found {target_filename} at: {full_path}")
                return full_path
    
    return None

def debug_codabench_structure():
    """Debug function to understand Codabench directory structure"""
    print("\n--- DEBUG: Scanning Codabench Directory Structure ---")
    root_paths = ['/app', '/app/ingested_program', '/app/program', '/app/data', '/app/output']
    
    for path in root_paths:
        if os.path.exists(path):
            print(f"\n📁 Contents of {path}:")
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        print(f"  📂 {item}/")
                    else:
                        print(f"  📄 {item}")
            except Exception as e:
                print(f"  Error reading {path}: {e}")
        else:
            print(f"\n❌ Path does not exist: {path}")
    print("--- End Debug Scan ---\n")

def main():
    print("=== INGESTION PHASE: CORE CODABENCH DATA PROCESSING ===")
    
    # Debug: Show current directory structure
    debug_codabench_structure()
    
    # ✅ FIXED: Your actual Gemini API key
    # IMPORTANT: Replace with your actual key
    GEMINI_API_KEY = ""  # <-- REPLACE THIS
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
        print("\n🚨 CRITICAL ERROR: Please add your actual Gemini API key!")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    # Setup output directory
    output_dir = '/app/output'
    submit_res_dir = os.path.join(output_dir, 'res')
    os.makedirs(submit_res_dir, exist_ok=True)
    
    TASKS = {
        'guj': 'guj.jsonl',
        'ben': 'ben.jsonl'
    }
    
    bridge_payload = {}
    files_found_count = 0
    
    for key, filename in TASKS.items():
        print(f"\n{'='*50}")
        print(f"Processing {key.upper()} track - looking for {filename}")
        print(f"{'='*50}")
        
        # Find the target file
        target_file = find_file_in_codabench_paths(filename)
        
        if target_file and os.path.exists(target_file):
            print(f"\n✅ SUCCESS: Found file for [{key}] at: {target_file}")
            files_found_count += 1
            
            # Determine the correct key based on language
            if key == "guj":
                text_key = "EN-HI-GU"
            else:  # ben
                text_key = "EN-HI-BN"
            
            bridge_payload[key] = []
            
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"📊 Total lines in file: {len(lines)}")
                    
                    for line_num, line in enumerate(lines, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            record = json.loads(line)
                            idx = record.get("index", line_num)
                            
                            # Try different possible text field names
                            text_str = (record.get(text_key) or 
                                      record.get("text") or 
                                      record.get("sentence"))
                            
                            if text_str:
                                print(f"\n  📝 Processing sample {idx}:")
                                print(f"     Text: {text_str[:100]}...")
                                
                                # ✅ FIXED: Call Gemini API with correct endpoint
                                tags_output = call_gemini_api(text_str, GEMINI_API_KEY)
                                
                                if not tags_output:
                                    print(f"     ⚠️ API failed, using fallback heuristic")
                                    # Simple fallback heuristic
                                    tokens = text_str.split()
                                    tagged_tokens = []
                                    for token in tokens:
                                        # Check for Unicode ranges
                                        if any('\u0900' <= c <= '\u097F' for c in token):
                                            tagged_tokens.append("HIN")
                                        elif any('\u0A80' <= c <= '\u0AFF' for c in token):
                                            tagged_tokens.append("GUJ")
                                        elif any('\u0980' <= c <= '\u09FF' for c in token):
                                            tagged_tokens.append("BEN")
                                        elif token in ['.', ',', '!', '?', ';', ':', '(', ')']:
                                            tagged_tokens.append("UNI")
                                        else:
                                            tagged_tokens.append("ENG")
                                    tags_output = " ".join(tagged_tokens)
                                
                                bridge_payload[key].append({
                                    "index": idx,
                                    "predicted_tags": tags_output
                                })
                                print(f"     ✅ Tags: {tags_output[:100]}...")
                            else:
                                print(f"     ⚠️ No text field found in entry {idx}")
                                
                        except json.JSONDecodeError as e:
                            print(f"     ❌ Invalid JSON: {e}")
                            continue
                            
            except Exception as e:
                print(f"❌ Error reading file: {e}")
        else:
            print(f"\n❌ ERROR: Could not find {filename}")
    
    # Check if any files were found
    if files_found_count == 0:
        print("\n🚨 CRITICAL FAILURE: No input files found!")
        print("Expected files: guj.jsonl and ben.jsonl")
        print("Based on debug output, your files are in: /app/ingested_program/")
        sys.exit(1)
    
    # Check if any entries were processed
    if not bridge_payload or all(len(v) == 0 for v in bridge_payload.values()):
        print("\n🚨 CRITICAL FAILURE: Files found but no entries processed!")
        sys.exit(1)
    
    # Save bridge file for scoring phase
    bridge_file_path = os.path.join(submit_res_dir, "ingested_payload.json")
    with open(bridge_file_path, "w", encoding="utf-8") as out_f:
        json.dump(bridge_payload, out_f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✅ INGESTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Files processed: {files_found_count}/2")
    print(f"Total entries: {sum(len(v) for v in bridge_payload.values())}")
    print(f"Output saved to: {bridge_file_path}")

if __name__ == "__main__":
    main()