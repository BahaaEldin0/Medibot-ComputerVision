import os
import random
import string
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures


# Define the effects and their mappings
from transforms.constants import (
    moire_mapping, blur_mapping, motion_mapping, glare_matte_mapping,
    glare_glossy_mapping, tilt_mapping, brightness_up_mapping, 
    brightness_down_mapping, contrast_up_mapping, contrast_down_mapping,
    identity_mapping, random_digital_mapping, rotation_mapping, 
    translation_mapping, exposure_mapping
)


# Define constants
COL_PATH = 'Path'
EFFECTS = {
    'moire': moire_mapping,
    'blur': blur_mapping,
    'motion': motion_mapping,
    'glare_matte': glare_matte_mapping,
    'glare_glossy': glare_glossy_mapping,
    'tilt': tilt_mapping,
    'brightness_up': brightness_up_mapping,
    'brightness_down': brightness_down_mapping,
    'contrast_up': contrast_up_mapping,
    'contrast_down': contrast_down_mapping,
    'identity': identity_mapping,
    'random-digital': random_digital_mapping,
    'rotation': rotation_mapping,
    'translation': translation_mapping,
    'exposure': exposure_mapping
}


# Helper function to generate new image name
def generate_new_name(effects_applied, original_name):
    prefix = ''.join([effect[0] for effect in effects_applied])
    return f"{prefix}{original_name}"

# Helper function to apply effects
def apply_effects(effects, src_img):
    for effect in effects:
        src_img = EFFECTS[effect](level=1, src_img=src_img)  # Assuming level 1 for simplicity
    return src_img

# Function to process each image
def process_image(path, effects, split, perturbed_dir):
    src_img = Image.open(path)
    dst_img = apply_effects(effects, src_img)
    new_name = generate_new_name(effects, path.name)
    dst_path = perturbed_dir / new_name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_img.save(dst_path)
    return str(dst_path).replace("\\", "/")

def generate_data(split, csv_path, perturbed_dir):
    df = pd.read_csv(csv_path)
    paths = list(df[COL_PATH])
    selected_paths = random.sample(paths, int(len(paths) * 0.7))
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for path in selected_paths:
            effects = random.sample(EFFECTS.keys(), random.randint(1, 5))  # Apply 1-5 random effects
            futures.append(executor.submit(process_image, Path(path), effects, split, perturbed_dir))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    
    new_df = pd.DataFrame(results, columns=[COL_PATH])
    new_df.to_csv(perturbed_dir / f'{split}.csv', index=False)


if __name__ == '__main__':
    np.random.seed(0)

    base_dir = Path("A:/Univeristy/Projects/Graduation-Project/Images Model/NIH Chest-Xrays")
    # splits = ['train', 'valid', 'test']
    splits = ['test']

    for split in splits:
        csv_path = base_dir / f"{split}.csv"
        perturbed_dir = base_dir / "Perturbed" / split
        perturbed_dir.mkdir(parents=True, exist_ok=True)
        generate_data(split, csv_path, perturbed_dir)

    print("Perturbed datasets have been generated successfully.")

