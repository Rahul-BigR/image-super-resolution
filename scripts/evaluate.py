import torch
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.srcnn_model import SRCNN
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_path, input_img_path, output_img_path):
    # Convert paths to absolute paths
    model_path = os.path.abspath('D:/Development/image-super-resolution/models/srcnn_model.pth')
    input_img_path = os.path.abspath('D:/Development/image-super-resolution/data/processed_images/LR_img1.jpg')
    output_img_path = os.path.abspath('D:/Development/image-super-resolution/data/processed_images/HR_img1.jpg')
    
    # Initialize model
    model = SRCNN().to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(input_img_path).convert('RGB')
    img = img.resize((33, 33), Image.BICUBIC)  # Ensure image size matches model input
    img = np.array(img).astype(np.float32) / 255.0  # Normalize image
    img = np.transpose(img, (2, 0, 1))  # Change to CxHxW format
    img = torch.tensor(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(img).squeeze(0).cpu().clamp(0, 1).numpy()  # Clamp values to [0, 1]
    
    # Post-process and save output image
    output_img = (np.transpose(output, (1, 2, 0)) * 255).astype(np.uint8)  # Convert to HxWxC
    output_img = Image.fromarray(output_img)
    output_img.save(output_img_path)

if __name__ == "__main__":
    model_path = '../models/srcnn_model.pth'
    input_img_path = '../data/test_images/input_image.png'
    output_img_path = '../data/test_images/output_image.png'
    evaluate_model(model_path, input_img_path, output_img_path)
