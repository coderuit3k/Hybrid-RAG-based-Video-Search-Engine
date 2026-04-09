import os
import sys
import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
import open_clip

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
ROOT_OCR = "data/MERGED"
OUTPUT_EMBED = "data/embed_ocr"
BATCH_SIZE = 64
DEVICE = 'cuda' 

# Model configuration
MODEL_NAME = 'xlm-roberta-base-ViT-B-32'
PRETRAINED = 'laion5b_s13b_b90k'


def clean_text(text):
    """
    Clean OCR text by removing special characters and normalizing whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    text = str(text).replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[|*_~=>\-]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_model():
    """
    Load OpenCLIP model and tokenizer
    
    Returns:
        tuple: (model, preprocess, tokenizer)
    """
    print(f"Loading OpenCLIP ({MODEL_NAME}) on {DEVICE}...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, 
            pretrained=PRETRAINED
        )
        model = model.to(DEVICE)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()
        print("Model loaded successfully!")
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def process_video_ocr(model, tokenizer, csv_file):
    """
    Process OCR embeddings for a single video
    
    Args:
        model: OpenCLIP model
        tokenizer: OpenCLIP tokenizer
        csv_file: CSV filename
    """
    video_id = csv_file.replace('.csv', '')
    
    # Load OCR data
    try:
        df = pd.read_csv(os.path.join(ROOT_OCR, csv_file))
    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return
    
    if 'ocr_text' not in df.columns or 'frame_idx' not in df.columns:
        print(f"Warning: CSV {csv_file} missing required columns")
        return

    # Group text by frame_idx
    df_grouped = df.groupby('frame_idx')['ocr_text'].apply(
        lambda x: " ".join([str(t) for t in x if str(t) != 'nan'])
    ).reset_index()
    
    frame_indices = df_grouped['frame_idx'].tolist()
    raw_texts = [clean_text(t) for t in df_grouped['ocr_text'].tolist()]
    processed_texts = [t if t else "Empty" for t in raw_texts]
    
    # Embed in batches
    for i in range(0, len(processed_texts), BATCH_SIZE):
        batch_txt = processed_texts[i : i + BATCH_SIZE]
        batch_frame_idx = frame_indices[i : i + BATCH_SIZE]
        
        try:
            # Tokenize & Encode
            text_tokens = tokenizer(batch_txt).to(DEVICE)
            with torch.no_grad():
                embeddings = model.encode_text(text_tokens)
                # Normalize
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
            
            embeddings = embeddings.cpu().numpy()
            
            # Save embeddings
            for j, vec in enumerate(embeddings):
                frame_idx = batch_frame_idx[j]
                try:
                    save_path = os.path.join(OUTPUT_EMBED, f"{video_id}_{frame_idx}.npy")
                    if os.path.exists(save_path):
                        print(f'Already exists {save_path}')
                        continue
                    
                    np.save(save_path, vec)
                    print(f'Done embedding and saved at {save_path}')
                except Exception as e:
                    print(f"Warning: Could not save embedding for {frame_idx}: {e}")
                    continue
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue


def main():
    """Main function to process all OCR files"""
    # Create output directory
    os.makedirs(OUTPUT_EMBED, exist_ok=True)
    
    # Validate input directory
    if not os.path.exists(ROOT_OCR):
        print(f"Error: OCR directory not found: {ROOT_OCR}")
        return
    
    # Load model
    try:
        model, _, tokenizer = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
        
    csv_files = sorted([f for f in os.listdir(ROOT_OCR) if f.endswith('.csv')])
    print(csv_files)
        
    for csv_file in tqdm(csv_files, leave=False):
        try:
            process_video_ocr(model, tokenizer, csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

        print(f'Done embedding {csv_file}')
    
    print("\nOCR embedding completed!")


if __name__ == "__main__":
    main()