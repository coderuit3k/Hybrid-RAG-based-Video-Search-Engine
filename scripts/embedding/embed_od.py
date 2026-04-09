import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
ROOT_OD = "data/MERGED"
OUTPUT_PATH = "data/embed_od"
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
MODEL_ID = 'openai/clip-vit-base-patch32'


def load_model():
    """
    Load CLIP model and processor
    
    Returns:
        tuple: (model, processor)
    """
    print(f'Loading CLIP model ({MODEL_ID}) on {DEVICE}...')
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model.eval()
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def embed_od(model, processor, text_list):
    """
    Generate embeddings for object detection text
    
    Args:
        model: CLIP model
        processor: CLIP processor
        text_list: List of text strings to embed
        
    Returns:
        numpy array of embeddings
    """
    try:
        # Process text with truncation for long texts (>77 tokens)
        inputs = processor(
            text=text_list, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        
        # Normalize vectors
        features /= features.norm(p=2, dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error in embed_od: {e}")
        raise


def save_embeddings(features_np, save_paths):
    """
    Save embeddings to disk
    
    Args:
        features_np: numpy array of embeddings
        save_paths: list of file paths to save to
    """
    for i, vector in enumerate(features_np):
        save_path = save_paths[i]
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, vector)
        except Exception as e:
            print(f"Warning: Could not save embedding to {save_path}: {e}")


def process_csv_file(model, processor, csv_file_path):
    """
    Process a single CSV file and generate embeddings
    
    Args:
        model: CLIP model
        processor: CLIP processor
        csv_file_path: Path to CSV file
    """
    try:
        od_df = pd.read_csv(csv_file_path)
        
        # Validate required columns
        required_cols = ['video_id', 'frame_idx', 'detected_objects']
        if not all(col in od_df.columns for col in required_cols):
            print(f"Warning: CSV {csv_file_path} missing required columns")
            return
        
        video_ids = od_df['video_id'].astype(str).tolist()
        frame_ids = od_df['frame_idx'].astype(str).tolist()
        detected_objects = od_df['detected_objects'].astype(str).tolist()
        
        batch_ods = []
        batch_paths = []
        
        for i in range(len(video_ids)):
            batch_ods.append(detected_objects[i])
            path = os.path.join(OUTPUT_PATH, f'{video_ids[i]}_{frame_ids[i]}.npy')
            batch_paths.append(path)
            
            # Process batch when full
            if len(batch_ods) >= BATCH_SIZE:
                try:
                    features_np = embed_od(model, processor, batch_ods)
                    save_embeddings(features_np, batch_paths)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                
                batch_ods = []
                batch_paths = []
        
        # Process remaining items
        if batch_ods:
            try:
                features_np = embed_od(model, processor, batch_ods)
                save_embeddings(features_np, batch_paths)
            except Exception as e:
                print(f"Error processing final batch: {e}")
                
    except Exception as e:
        print(f"Error processing CSV file {csv_file_path}: {e}")


def main():
    """Main function to process all OD files"""
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Validate input directory
    if not os.path.exists(ROOT_OD):
        print(f"Error: OD directory not found: {ROOT_OD}")
        return
    
    # Load model
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get CSV files
    csv_files = sorted([f for f in os.listdir(ROOT_OD) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"No CSV files found in {ROOT_OD}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc='Embedding OD'):
        csv_file_path = os.path.join(ROOT_OD, csv_file)
        try:
            process_csv_file(model, processor, csv_file_path)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print("\nOD embedding completed!")


if __name__ == '__main__':
    main()