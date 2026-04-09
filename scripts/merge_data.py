import os
import pandas as pd
import re
import glob

# Paths
OD_DIR = os.path.join("data", "OD")
OCR_DIR = os.path.join("data", "OCR")
MERGED_DIR = os.path.join("data", "MERGED")

os.makedirs(MERGED_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove URL-like patterns
    text = re.sub(r'http\S+', '', text)
    
    # Remove OCR boilerplate (case-insensitive)
    boilerplate = [
        r"chỉ trích xuất nguyên văn.*",
        r"không được thêm, sửa.*",
        r"nếu chữ bị mờ.*",
        r"trích xuất văn bản.*",
        r"không đọc được"
    ]
    for phrase in boilerplate:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # Replace non-word characters (preserving Vietnamese accents) with spaces.
    # \w in Python 3 matches [a-zA-Z0-9_] plus Unicode characters (including Vietnamese accents).
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (replace multiple spaces/newlines with single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()


def read_ocr_csv(ocr_file):
    """
    Robustly reads an OCR CSV file, detecting multiline text records
    and flattening them into a single line per filename.
    Equivalent to fix_ocr_csv.py logic but returns a DataFrame.
    """
    if not os.path.exists(ocr_file):
        return None
    
    # Manual parsing fallback
    with open(ocr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        return None

    records = []
    current_filename = None
    current_text_buffer = []

    # Skip header if present
    start_index = 0
    if lines[0].strip().startswith("filename,"):
        start_index = 1
    
    for i in range(start_index, len(lines)):
        line = lines[i].rstrip('\n').rstrip('\r') 
        
        # Heuristic: Valid new record starts with 4 digits + .jpg + comma
        match = re.match(r'^(\d+\.jpg),', line)
        
        if match:
            # Save previous record
            if current_filename:
                full_text = " ".join(current_text_buffer)
                
                # Manual CSV quoting cleanup
                if len(full_text) >= 2 and full_text.startswith('"') and full_text.endswith('"'):
                    content = full_text[1:-1].replace('""', '"')
                else:
                    content = full_text
                    
                records.append({'filename': current_filename, 'ocr_text': content})
            
            # Start new record
            current_filename = match.group(1)
            text_part = line[len(current_filename)+1:]
            current_text_buffer = [text_part]
        
        else:
            # Continuation
            if current_filename:
                current_text_buffer.append(line)
    
    # Save last record
    if current_filename:
        full_text = " ".join(current_text_buffer)
        if len(full_text) >= 2 and full_text.startswith('"') and full_text.endswith('"'):
            content = full_text[1:-1].replace('""', '"')
        else:
            content = full_text
        records.append({'filename': current_filename, 'ocr_text': content})

    if not records:
        return None
        
    return pd.DataFrame(records)

def process_files():
    print(f"Checking for OD files in: {OD_DIR}")
    od_files = glob.glob(os.path.join(OD_DIR, "*.csv"))
    
    if not od_files:
        print("No OD files found.")
        return

    print(f"Found {len(od_files)} OD files. Starting merge...")

    # Keyframes directory (source of truth for frame_idx)
    KEYFRAMES_DIR = os.path.join(BASE_DIR, "data", "keyframes")

    for od_file in od_files:
        video_id = os.path.splitext(os.path.basename(od_file))[0]
        
        try:
            df_od = pd.read_csv(od_file)
        except Exception as e:
            print(f"Error reading OD file {od_file}: {e}")
            continue

        # Create filename column for merging
        if 'image_path' in df_od.columns:
            # Extract filename from path (e.g. c:\...\0001.jpg -> 0001.jpg)
            df_od['filename'] = df_od['image_path'].apply(lambda x: os.path.basename(str(x)))
            
            # Extract 'n' from filename for joining with keyframes CSV
            # '0001.jpg' -> 1
            def extract_n(filename):
                try:
                    name_part = os.path.splitext(filename)[0]
                    return int(name_part)
                except:
                    return None
            
            df_od['n'] = df_od['filename'].apply(extract_n)
        else:
            print(f"Warning: 'image_path' column missing in {od_file}. Skipping.")
            continue
        
        # --- Merge with Keyframes CSV (for correct frame_idx) ---
        # Video ID format: L01_V001 -> Folder L01
        try:
            folder_name = video_id.split('_')[0] # 'L01'
            kf_csv_path = os.path.join(KEYFRAMES_DIR, folder_name, f"{video_id}.csv")
            
            if os.path.exists(kf_csv_path):
                df_kf = pd.read_csv(kf_csv_path)
                # Ensure 'n' is mapped to correct frame_idx
                # Keyframe CSV columns: n, pts_time, fps, frame_idx
                if 'n' in df_kf.columns and 'frame_idx' in df_kf.columns:
                    # Merge on 'n'
                    df_od = pd.merge(df_od, df_kf[['n', 'frame_idx']], on='n', how='left', suffixes=('_old', ''))
                    
                    # If frame_idx was already in OD, drop it (it's likely wrong/sequential)
                    if 'frame_idx_old' in df_od.columns:
                        df_od.drop(columns=['frame_idx_old'], inplace=True)
                else:
                    print(f"Warning: Keyframe CSV for {video_id} missing columns 'n' or 'frame_idx'")
            else:
                print(f"Warning: Keyframe CSV not found at {kf_csv_path}")
        except Exception as e:
            print(f"Error processing keyframe CSV for {video_id}: {e}")

        
        # --- Merge with OCR CSV ---
        ocr_file = os.path.join(OCR_DIR, f"{video_id}.csv")
        
        # Use new robust reading function
        df_ocr = read_ocr_csv(ocr_file)
        
        if df_ocr is not None:
            # Ensure filename is string for merge
            df_ocr['filename'] = df_ocr['filename'].astype(str)
            df_od['filename'] = df_od['filename'].astype(str)
            
            if 'ocr_text' in df_ocr.columns:
                df_merged = pd.merge(df_od, df_ocr[['filename', 'ocr_text']], on='filename', how='left')
            else:
                df_merged = df_od.copy()
                df_merged['ocr_text'] = ""
        else:
            # No OCR data, just use OD
            df_merged = df_od.copy()
            df_merged['ocr_text'] = ""

        # Fill NaNs
        df_merged['ocr_text'] = df_merged['ocr_text'].fillna("")
        df_merged['detected_objects'] = df_merged['detected_objects'].fillna("")
        
        # Apply cleaning
        df_merged['ocr_text'] = df_merged['ocr_text'].apply(clean_text)
        df_merged['detected_objects'] = df_merged['detected_objects'].apply(clean_text)
        
        # Clean up temporary columns
        cols_to_drop = ['n', 'filename', 'object_counts', 'top_10_scores', 'raw_count']
        df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns], inplace=True)

        # Save to MERGED directory
        output_path = os.path.join(MERGED_DIR, f"{video_id}.csv")
        df_merged.to_csv(output_path, index=False)
        # print(f"Saved merged: {output_path}")

    print(f"Merge completed. Files saved to {MERGED_DIR}")

if __name__ == "__main__":
    process_files()

