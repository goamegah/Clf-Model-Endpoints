import requests # type: ignore
import io
import json
import os
import base64

import torch
from PIL import Image
import torchvision

from typing import List

import numpy as np

from mlserver import MLModel
from mlserver.codecs import decode_args

import warnings

from typing import List

from resnet import ResNet18, BasicBlock

from pathlib import Path

PATH = Path(__file__).parent

# Load configuration from JSON file
with open(f'{PATH}/config_ckpt.json', 'r') as f:
    config = json.load(f)

model_urls = config['model_urls']

GRAYSCALE = True  # for MNIST dataset
NUM_CLASSES = 10

def load_checkpoint(model_path, device, out='checkpoint.pth.tar'):
    # URL publique pour télécharger les poids

    ckpt_path = f'./tmp/{out}'


    # Télécharger le fichier des poids
    response = requests.get(model_path)

    # Sauvegarder le fichier des poids
    if response.status_code == 200:
        os.makedirs('./tmp', exist_ok=True)
        with open(ckpt_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")  
    
    # Charger les poids dans le modèle si le téléchargement a réussi
    if os.path.exists(ckpt_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                checkpoint = torch.load(ckpt_path, map_location=device)  # Remplacez 'cpu' par 'device' si nécessaire
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print("Model weights file does not exist.")
    
    # # remove the file
    os.remove(ckpt_path)
    os.removedirs('./tmp')
    return checkpoint


def read_image_as_pil(encoded_image:str):
    pil_image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    return pil_image


def load_model():    
    model = ResNet18(num_layers=18, block=BasicBlock, 
                     num_classes=10, grayscale=GRAYSCALE)
    
    checkpoint_uri = model_urls['resnet18_train_0100']
    ckpt = load_checkpoint(model_path=checkpoint_uri, device='cpu')
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=False)
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


class MLServerResnet(MLModel):

    # Load the model into memory
    async def load(self) -> bool:
        self._model = load_model()
        self.ready = True
        return self.ready

    @decode_args
    async def predict(self, payload: List[str]) -> np.ndarray:
        pil_images = [read_image_as_pil(p) for p in payload]

        # process image step
        images_tensor = [process_pilimage_to_tensor(pil) for pil in pil_images]
    
        images_tensor = torch.stack(images_tensor).squeeze(1)
        
        self._model.eval()
        self._model = self._model.to(device='cpu')

        # Inference step with torch.no_grad to avoid tracking gradients
        with torch.no_grad():
            logits = self._model(images_tensor)
            predictions = torch.argmax(input=logits, dim=1)

        return predictions.numpy()