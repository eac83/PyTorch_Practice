"""
Uses a trained PyTorch model to make a prediction on an image.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import data_setup, model_builder

from pathlib import Path
from PIL import Image
from torchvision import transforms
from typing import List

# Setup parser
parser = argparse.ArgumentParser(
    prog='predict.py',
    description='Uses a trained PyTorch model to make a prediction on an image.'
)
parser.add_argument('image')
args = parser.parse_args()
image_path = Path('data') / args.image

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available else 'cpu'

# Grab class names
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_data = torchvision.datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

class_names = train_data.classes

# Load model
MODEL_SAVE_PATH = Path('models/05_going_modular_script_mode_tinyvgg_model.pth')
kwargs, state = torch.load(MODEL_SAVE_PATH)
model = model_builder.TinyVGG(**kwargs)
model.load_state_dict(state)

# Load in image and convert the tensor values to float32
target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
original_image = torch.clone(target_image)
original_image = np.array(Image.open(image_path))

# Divide the image pixel values by 255 to get them between [0, 1]
target_image = target_image / 255.

# Create transform pipeline to resize image
transform = transforms.Compose([
    transforms.Resize((64, 64))
])
target_image = transform(target_image)

# Make sure the model is on the target device
model.to(device)

# Turn on model evaluation mode and inference mode
model.eval()
with torch.inference_mode():
    # Add an extra dimension to the image
    target_image = target_image.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension and send it to the target device
    target_image_pred = model(target_image.to(device))

# Convert logits -> prediction probabilities
target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

# Covert prediction probabilities -> prediction labels
target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

# Plot the image alongside the prediction and prediction probability
plt.imshow(original_image) # make sure it's the right size for Matplotlib
if class_names:
    title = f'Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}'
else:
    title = f'Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}'
plt.title(title)
plt.axis(False)
plt.show()
print(f'Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}')
