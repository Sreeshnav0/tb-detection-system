import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def get_gradcam(model, input_tensor, target_layer, class_idx=None):
    """
    Computes Grad-CAM heatmap for a given input and target layer.
    """
    # Dictionary to store activations and gradients
    feature_maps = {}
    gradients = {}

    def save_feature_map(module, input, output):
        feature_maps['value'] = output

    def save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Register hooks
    handle_forward = target_layer.register_forward_hook(save_feature_map)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(input_tensor)
    
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    
    # Backward pass
    model.zero_grad()
    score = output[0, class_idx]
    score.backward()

    # Get Captured values
    grads = gradients['value']
    fmaps = feature_maps['value']

    # Weighting the feature maps by the mean of the gradients
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * fmaps, dim=1).squeeze()

    # Apply ReLU to keep only positive contributions
    grad_cam = F.relu(grad_cam)

    # Normalize heatmap
    grad_cam = grad_cam.detach().cpu().numpy()
    grad_cam = (grad_cam - np.min(grad_cam)) / (np.max(grad_cam) - np.min(grad_cam) + 1e-8)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return grad_cam, class_idx, F.softmax(output, dim=1)[0, class_idx].item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1-3. Load model architecture
    model = models.resnet50(pretrained=False) # Architecture only
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # 4. Load weights
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 5. Set to evaluation mode
    model.eval()

    # Automatically pick a sample from dataset/test/Tuberculosis
    tb_dir = os.path.join('dataset', 'test', 'Tuberculosis')
    if not os.path.exists(tb_dir):
        print(f"Error: {tb_dir} not found.")
        return
    
    samples = [f for f in os.listdir(tb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not samples:
        print("No images found in TB test directory.")
        return
    
    img_name = random.choice(samples)
    img_path = os.path.join(tb_dir, img_name)
    print(f"Processing sample: {img_path}")

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform image
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 6. Hook into last convolutional layer (layer4)
    target_layer = model.layer4

    # 7-9. Compute Grad-CAM
    heatmap, pred_idx, prob = get_gradcam(model, input_tensor, target_layer)

    classes = ['Normal', 'Tuberculosis']
    print(f"Predicted class: {classes[pred_idx]}")
    print(f"Prediction probability: {prob:.4f}")

    # 10. Overlay heatmap using OpenCV
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224)) # Match the crop/resize for overlay logic
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay logic: combined = alpha * original + (1-alpha) * heatmap
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # 11. Save output
    output_path = 'gradcam_output.jpg'
    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM output saved to {output_path}")

if __name__ == "__main__":
    main()
