import os
import sys
import subprocess
import importlib

# Required libraries check
required_libraries = ["gliner", "pandas", "openpyxl", "torch"]

def check_and_install_libraries():
    """Checks if libraries are installed and installs them if they are missing."""
    for lib in required_libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"Library '{lib}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

check_and_install_libraries()

import glob
import shutil
import pandas as pd
import torch
import re
from gliner import GLiNER
from collections import Counter

# User configuration

# Specify the path to your folder (e.g., "/Users/name/Desktop/folder")
drive_folder_path = "/Users/Folder" 

# List of words or labels that should be ignored during extraction
ignore_list = ["example", "example2", "example3"]

# If set to True, the script will only analyze lines that do not end with '?' or start with interviewer prefixes
only_answers_mode = True 

# Definition of entities to find. Define column names, descriptions (if necessary) and color
entity_config = [
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#ff9933", "memo": "Your description"}, # Use HEX code in the https://colorkit.co/color/99ccff/ in order to find the color you need
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#ff9999", "memo": "Your description"},
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#99ff99", "memo": "gYour description"},
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#99ccff", "memo": "Your description"},
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#c2c2f0", "memo": "Your description"},
    {"label": "Your_Label", "column_name": "Your_column_name", "color": "#ffff99", "memo": "Your description"}
]

# Main script logic

# Device selection for local processing
# Hardware acceleration setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using computing device: {device.upper()}")

# Load the zero-shot NER model
print("Loading GLiNER model...")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)

labels = [item["label"] for item in entity_config if item["label"] not in ignore_list]
color_map = {item["label"]: item["color"] for item in entity_config}
memo_map = {item["label"]: item["memo"] for item in entity_config}

def is_question(text):
    """Checks if a line is a question."""
    text = text.strip()
    if text.endswith('?') or re.match(r'^(q|f|i|interviewer|frage):', text, re.IGNORECASE):
        return True
    return False

def process_long_text(text, labels, chunk_size=1000, overlap=150):
    """Processes text chunks and handles answer-only filtering."""
    if only_answers_mode:
        lines = text.split('\n')
        filtered_blocks = []
        current_offset = 0
        for line in lines:
            line_len = len(line) + 1 
            if not is_question(line):
                filtered_blocks.append((line, current_offset))
            current_offset += line_len
            
        all_entities = []
        for block_text, block_offset in filtered_blocks:
            if len(block_text.strip()) < 2: continue
            # Increased model sensitivity to improve the detection of German compound nouns such as Rockmusik or Jazzmusik
            block_entities = model.predict_entities(block_text, labels, threshold=0.20)
            for ent in block_entities:
                ent['start'] += block_offset
                ent['end'] += block_offset
                all_entities.append(ent)
    else:
        all_entities = []
        start_offset = 0
        while start_offset < len(text):
            end_offset = start_offset + chunk_size
            chunk = text[start_offset:end_offset]
            chunk_entities = model.predict_entities(chunk, labels, threshold=0.20)
            for ent in chunk_entities:
                ent['start'] += start_offset
                ent['end'] += start_offset
                all_entities.append(ent)
            start_offset += (chunk_size - overlap)

    unique_entities = []
    seen = set()
    ignore_set = {w.lower() for w in ignore_list}
    for ent in sorted(all_entities, key=lambda x: x['start']):
        pos = (ent['start'], ent['end'], ent['label'])
        word_clean = ent['text'].strip().lower()
        if pos not in seen and word_clean not in ignore_set:
            unique_entities.append(ent)
            seen.add(pos)
    return unique_entities

def generate_html(text, entities, filename):
    """Creates html where counters are in the legend without additional headers."""
    entities = sorted(entities, key=lambda x: x['start'])
    label_counts = Counter([e['label'] for e in entities])

    html_body = ""
    last_idx = 0
    for entity in entities:
        start = entity['start']
        end = entity['end']
        label = entity['label']
        word = text[start:end]
        
        html_body += text[last_idx:start]
        color = color_map.get(label, "#cccccc")
        html_body += f"<span style='background-color: {color}; padding: 2px; border-radius: 3px; font-weight: bold; border: 1px solid #888;' title='{label}'>{word}</span>"
        last_idx = end

    html_body += text[last_idx:]
    html_body = html_body.replace("\n", "<br>")

    legend_items = []
    for item in entity_config:
        count = label_counts.get(item['label'], 0)
        legend_items.append(f"<span style='background-color: {item['color']}; padding: 2px 5px; margin-right: 5px;'>{item['label']} ({count})</span>")

    full_html = f"""
    <html>
    <head><meta charset="utf-8"><title>{filename}</title></head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: auto;">
        <h2>analysis: {filename}</h2>
        <div style="padding: 15px; background: #f0f0f0; border-radius: 5px; margin-bottom: 20px;">
            {" ".join(legend_items)}
            {f"<br><small><i>mode: only answers enabled</i></small>" if only_answers_mode else ""}
        </div>
        <div style="white-space: pre-wrap;">{html_body}</div>
    </body>
    </html>
    """
    return full_html

# Main execution loop

if not os.path.exists(drive_folder_path):
    print(f"Error: Path '{drive_folder_path}' not found.")
else:
    temp_html_dir = "temp_html_files"
    final_dir = "analysis_results"
    for p in [temp_html_dir, final_dir]:
        if os.path.exists(p): shutil.rmtree(p)
        os.makedirs(p)

    txt_files = glob.glob(os.path.join(drive_folder_path, "*.txt"))
    table_data = []
    unique_codes_map = {}

    print(f"Processing {len(txt_files)} files...")

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            entities = process_long_text(text, labels)

            for entity in entities:
                code_path = f"{entity['label']}\\{entity['text'].strip()}"
                if code_path not in unique_codes_map: unique_codes_map[code_path] = set()
                unique_codes_map[code_path].add(filename)

            row = {"filename": filename}
            for config in entity_config:
                matches = [e['text'] for e in entities if e['label'] == config["label"]]
                row[config["column_name"]] = "; ".join(sorted(list(set(matches)))) if matches else ""
            table_data.append(row)

            html_content = generate_html(text, entities, filename)
            with open(os.path.join(temp_html_dir, f"{filename}.html"), "w", encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
# Result export section
    if table_data:
        maxqda_data = []
        top_level_codes = {item["label"]: set() for item in entity_config}
        for code_path, filenames in unique_codes_map.items():
            maxqda_data.append({"code": code_path, "memo": "", "found_in_files": "; ".join(sorted(list(filenames)))})
            main_label = code_path.split('\\')[0]
            if main_label in top_level_codes: top_level_codes[main_label].update(filenames)
        for label, files_found in top_level_codes.items():
            if files_found:
                 maxqda_data.insert(0, {"code": label, "memo": memo_map.get(label, ""), "found_in_files": "; ".join(sorted(list(files_found)))})
        
        pd.DataFrame(maxqda_data).to_excel(os.path.join(final_dir, "maxqda_import_codes.xlsx"), index=False)
        pd.DataFrame(table_data).to_excel(os.path.join(final_dir, "summary_data.xlsx"), index=False)
        pd.DataFrame(table_data).to_csv(os.path.join(final_dir, "summary_data.csv"), index=False, sep=";")
        
        shutil.make_archive(os.path.join(final_dir, "coded_html_files"), 'zip', root_dir=temp_html_dir)
        shutil.rmtree(temp_html_dir)
        print(f"Done. Results saved in '{final_dir}'.")