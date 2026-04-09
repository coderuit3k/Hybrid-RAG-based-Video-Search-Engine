import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
ROOT = "data/keyframes"
OUTPUT_PATH = "data/embed_kf"
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
MODEL_ID = 'openai/clip-vit-base-patch32'


def load_model():
    """
    Load CLIP model and processor
    
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading CLIP model ({MODEL_ID}) on {DEVICE}...")
    try:
        model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model.eval()
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def embed_kf(model, processor, images):
    """
    Generate embeddings for keyframe images
    
    Args:
        model: CLIP model
        processor: CLIP processor
        images: List of PIL Images
        
    Returns:
        numpy array of embeddings
    """
    try:
        inputs = processor(images=images, return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        
        # Normalize vectors
        features /= features.norm(p=2, dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error in embed_kf: {e}")
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


def process_video_keyframes(model, processor, kf_folder, csv_file, video_id):
    """
    Process keyframes for a single video
    
    Args:
        model: CLIP model
        processor: CLIP processor
        kf_folder: Path to keyframes folder
        csv_file: Path to CSV mapping file
        video_id: Video ID
    """
    # Load frame index mapping
    try:
        df = pd.read_csv(csv_file)
        if 'n' not in df.columns or 'frame_idx' not in df.columns:
            print(f"Warning: CSV {csv_file} missing required columns (n, frame_idx)")
            return
        mapping_dict = dict(zip(df['n'], df['frame_idx']))
    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return
    
    # Get image files
    image_files = sorted([
        f for f in os.listdir(kf_folder) 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not image_files:
        print(f"Warning: No images found in {kf_folder}")
        return
    
    print("-" * 60)
    print(f"Processing: {video_id} with {len(image_files)} frames")
    
    batch_imgs = []
    batch_paths = []
    
    # Process images
    for img_name in tqdm(image_files, desc=f"Video {video_id}", leave=False):
        try:
            # Parse image ID (e.g., "001.jpg" -> 1)
            n_id = int(img_name.split('.')[0])
            
            if n_id not in mapping_dict:
                continue
            
            frame_idx = mapping_dict[n_id]
            save_path = os.path.join(OUTPUT_PATH, f"{video_id}_{frame_idx}.npy")
            
            # Skip if already processed
            if os.path.exists(save_path):
                continue
            
            # Load image
            img_path = os.path.join(kf_folder, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue
            
            batch_imgs.append(img)
            batch_paths.append(save_path)
            
            # Process batch when full
            if len(batch_imgs) >= BATCH_SIZE:
                try:
                    features_np = embed_kf(model, processor, batch_imgs)
                    save_embeddings(features_np, batch_paths)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                
                batch_imgs = []
                batch_paths = []
                
        except ValueError:
            # Skip files that don't match expected format
            continue
        except Exception as e:
            print(f"Error processing frame {img_name}: {e}")
            continue
    
    # Process remaining images
    if batch_imgs:
        try:
            features_np = embed_kf(model, processor, batch_imgs)
            save_embeddings(features_np, batch_paths)
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    print(f'Done embedding keyframes from {video_id}')


def main():
    """Main function to process all keyframes"""
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Validate input directory
    if not os.path.exists(ROOT):
        print(f"Error: Keyframes directory not found: {ROOT}")
        return
    
    # Load model
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Process each folder (L01, L02, etc.)
    folders = sorted([
        f for f in os.listdir(ROOT) 
        if os.path.isdir(os.path.join(ROOT, f))
    ])
    
    for folder in folders:
        folder_path = os.path.join(ROOT, folder)
        video_ids = sorted([
            v for v in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, v))
        ])
        
        for video_id in video_ids:
            kf_folder = os.path.join(folder_path, video_id)
            csv_file = os.path.join(folder_path, f'{video_id}.csv')
            
            if not os.path.exists(csv_file):
                print(f"Warning: CSV file not found: {csv_file}")
                continue
            
            try:
                process_video_keyframes(model, processor, kf_folder, csv_file, video_id)
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue
    
    print("\nKeyframe embedding completed!")


if __name__ == '__main__':
    main()