import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model(model_path='models/best_model.pth'):
    global model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def get_risk_level(prob):
    if prob > 0.80:
        return "High Risk"
    elif prob > 0.60:
        return "Moderate Risk"
    else:
        return "Low Risk"

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image).unsqueeze(0).to(device)
    return tensor

def generate_gradcam(input_tensor, image_path, output_path, target_class_idx=1):
    global model
    feature_maps = {}
    gradients = {}

    def save_feature_map(module, input, output):
        feature_maps['value'] = output

    def save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Target the last convolutional layer
    target_layer = model.layer4
    handle_forward = target_layer.register_forward_hook(save_feature_map)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    input_tensor.requires_grad = True
    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()
    
    # Backward pass targeting specific class (default index 1 for TB)
    model.zero_grad()
    score = output[0, target_class_idx]
    score.backward()

    # Grad-CAM calculation
    grads = gradients['value']
    fmaps = feature_maps['value']
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * fmaps, dim=1).squeeze()
    grad_cam = F.relu(grad_cam).detach().cpu().numpy()
    
    # Normalize
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + 1e-8)

    # Apply clinical threshold: remove weak activations below 0.4
    grad_cam[grad_cam < 0.4] = 0

    # Overlay
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_resized = cv2.resize(grad_cam, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    
    cv2.imwrite(output_path, overlay)

    # Cleanup
    handle_forward.remove()
    handle_backward.remove()
    
    return class_idx, F.softmax(output, dim=1)[0]
