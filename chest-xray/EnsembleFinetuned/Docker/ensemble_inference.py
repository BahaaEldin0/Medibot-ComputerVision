import os
import torch
from fastai.vision.all import *
from PIL import Image
from io import BytesIO
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import timm 


# Set the seed for reproducibility
SEED = 85
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# Define disease labels based on the original dataset
disease_labels = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# Load the new sample dataset
dataset = load_dataset('BahaaEldin0/ensembleModelsInference_dls_dataset_sample')
test_set = dataset['test']
test_df = test_set.to_pandas()

# Define disease labels based on the original dataset
disease_labels = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# One-hot encode the disease labels
for disease in tqdm(disease_labels): 
    test_df[disease] = test_df['label'].apply(lambda x: 1 if disease in x else 0)


# Define item transforms and batch transforms
item_transforms = [
    Resize((224, 224)),
]

batch_transforms = [
    Flip(),
    Rotate(),
    Normalize.from_stats(*imagenet_stats),
]

# Define get_x and get_y functions
def get_x(row):
    return PILImage.create(BytesIO(row['image']['bytes']))

def get_y(row):
    labels = row[disease_labels].tolist()
    return labels

test_df['image_bytes'] = test_df['image'].apply(lambda x: x['bytes'])

# Create DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock(encoded=True, vocab=disease_labels)),
    splitter=RandomSplitter(valid_pct=0.125, seed=SEED),
    get_x=get_x,
    get_y=get_y,
    item_tfms=item_transforms,
    batch_tfms=batch_transforms
)

# Create DataLoaders
dls = dblock.dataloaders(test_df, bs=1)

# Load the models
models = {
    'swinv2': vision_learner(dls, 'swinv2_cr_small_ns_224.sw_in1k', metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()]),
    'volod2': vision_learner(dls, 'volo_d2_224.sail_in1k', metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()]),
    'dense121': vision_learner(dls, models.densenet121, metrics=[accuracy_multi, F1ScoreMulti(), RocAucMulti()])
}

# Load model weights
models['swinv2'].load('../models/swin_finetuned')
models['volod2'].load('../models/volo2d_finetuned')
models['dense121'].load('../models/densenet121_finetuned')

# Define FastAPI app
app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    contents = await file.read()
    img = PILImage.create(BytesIO(contents))
    img = dls.after_item(img)
    img = img.to(torch.device('cpu'))
    img = dls.after_batch(img.unsqueeze(0))
    
    # Make predictions with all models
    results = {}
    for name, model in models.items():
        model.model.eval()
        with torch.no_grad():
            preds = model.model(img)
            probs = torch.sigmoid(preds).numpy()[0]
            predicted_labels = [disease_labels[i] for i, p in enumerate(probs) if p > 0.5]
            results[name] = {
                'predicted_labels': predicted_labels,
                'probabilities': probs.tolist()
            }
    
    # Ensemble predictions
    ensemble_probs = sum(torch.sigmoid(model.model(img)).detach().numpy()[0] for model in models.values()) / len(models)
    ensemble_predicted_labels = [disease_labels[i] for i, p in enumerate(ensemble_probs) if p > 0.5]
    results['ensemble'] = {
        'predicted_labels': ensemble_predicted_labels,
        'probabilities': ensemble_probs.tolist()
    }

    return JSONResponse(results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

