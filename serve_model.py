import requests # type: ignore
import io
import json
import os

from fastapi import FastAPI, UploadFile, File, Form
import torch
from PIL import Image
import torchvision


from typing import List

from model import ResNet18, BasicBlock

# Load configuration from JSON file
with open(f'./config_ckpt.json', 'r') as f:
    config = json.load(f)

model_urls = config['model_urls']

GRAYSCALE = True  # for MNIST dataset
NUM_CLASSES = 10

# instanciate API class
app = FastAPI()

def load_checkpoint(model_path, device, out='checkpoint.pth.tar'):
    # URL publique pour télécharger les poids

    ckpt_path = f'./tmp/{out}'

    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Télécharger le fichier des poids
    response = requests.get(model_path)

    # Vérifier si le téléchargement a réussi
    if response.status_code == 200:
        with open(ckpt_path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    # Charger les poids dans le modèle si le téléchargement a réussi
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)  # Remplacez 'cpu' par 'device' si nécessaire
            print("Model checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print("Model weights file does not exist.")
    
    # # remove the file
    os.remove(ckpt_path)
    os.removedirs('./tmp')

    return checkpoint

def read_image_as_pil(encoded_image):
    pil_image = Image.open(io.BytesIO(encoded_image))
    return pil_image

def load_model(name):

    model = None
    match name:
        case "ResNet18":    
            model = ResNet18(num_layers=18, block=BasicBlock, 
                             num_classes=10, grayscale=GRAYSCALE)
            checkpoint_uri = model_urls['resnet18_train_0100']
        case "LetNet5":    
            model = ...
            checkpoint_uri = ...
        case _:
            model = ...
            checkpoint_uri = ...

    ckpt = load_checkpoint(model_path=checkpoint_uri, device='cpu')
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # message to confirm the model is loaded
    print(f'Model {name} loaded successfully)')

    return model

def process_pilimage_to_tensor(image):
    # Resize image to 32x32
    img = image.resize((32, 32))

    # Apply transformation using torchvision modeul
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1), # to be sure to have one channel
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))  # Normalisation (mean and standard deviation)
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # add one dimension for batch dimension
    print(f'==+> {img_tensor.shape}')

    return img_tensor

@app.get(path="/")
def welcome():
    return {"message": "welcome API"}

@app.post("/prediction")
async def predict(files: List[UploadFile] = File(...), model_id: str = Form(...)):


    # get encoded image step
    contents = [await f.read() for f in files]
    pil_images = [read_image_as_pil(c) for c in contents]

    # process image step
    images_tensor = [process_pilimage_to_tensor(pil) for pil in pil_images]

    # model loading step
    model = load_model(model_id)

    model.eval()
    model = model.to(device='cpu')

    # Inference step with torch.no_grad to avoid tracking gradients
    with torch.no_grad():
        logits_l = [model(image_tensor) for image_tensor in images_tensor]
        predictions = [torch.argmax(input=logits) for logits in logits_l]

    return {"predictions": [p.item() for p in predictions]}