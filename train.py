import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import copy
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score
import time
import numpy as np
import torch.nn.functional as F

def main():
    # Data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Data directory
    data_dir = 'dataset'

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False),
    }

    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Classes: {class_names}")
    for x in ['train', 'val', 'test']:
        print(f"{x.capitalize()} dataset size: {dataset_sizes[x]}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet50
    model = models.resnet50(weights='DEFAULT')

    # Freeze first 70% of layers
    layers = list(model.children())
    num_layers = len(layers)
    freeze_until = int(num_layers * 0.7)
    
    print(f"Total layers: {num_layers}, freezing first {freeze_until} layers.")
    
    for i, child in enumerate(model.children()):
        if i < freeze_until:
            for param in child.parameters():
                param.requires_grad = False

    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Move model to device
    model = model.to(device)

    # Calculate class weights
    targets = image_datasets['train'].targets
    class_counts = [targets.count(i) for i in range(len(class_names))]
    num_samples = sum(class_counts)
    
    # Calculate weights: N / (C * n_i)
    weights = [num_samples / (len(class_names) * count) for count in class_counts]
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class counts: {dict(zip(class_names, class_counts))}")
    print(f"Calculated class weights: {dict(zip(class_names, weights))}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # Training loop
    num_epochs = 20
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.data.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            else:
                # Calculate recall for validation
                # Tuberculosis class is usually index 1
                val_recall = recall_score(all_labels, all_preds, pos_label=1)
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {val_recall:.4f}")

                # Deep copy the model if it's the best one so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save best model to disk
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    torch.save(best_model_wts, 'models/best_model.pth')
                    print(f"Best model saved with Accuracy: {best_acc:.4f}")

    print(f"\nTraining complete. Best Val Acc: {best_acc:4f}")

    # Evaluation on Test Set
    print("\n" + "="*30)
    print("Evaluating Best Model on Test Set")
    print("="*30)
    
    # Load the best model weights
    if os.path.exists('models/best_model.pth'):
        model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))
        print("Loaded best model from 'models/best_model.pth'")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # ROC-AUC (using probabilities for the positive class)
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    # Print baseline metrics
    print(f"\nTest Accuracy (Threshold 0.5): {test_acc:.4f}")
    print(f"ROC-AUC score:                {roc_auc:.4f}")
    print("\nConfusion Matrix (Threshold 0.5):")
    print(conf_matrix)
    
    # Threshold Analysis
    print("\n" + "="*30)
    print("Threshold Analysis (TB Class)")
    print("="*30)
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
    print("-" * 45)
    
    thresholds = [0.5, 0.6, 0.7, 0.8]
    for t in thresholds:
        t_preds = (all_probs[:, 1] >= t).astype(int)
        t_precision = precision_score(all_labels, t_preds, zero_division=0)
        t_recall = recall_score(all_labels, t_preds, zero_division=0)
        t_f1 = f1_score(all_labels, t_preds, zero_division=0)
        print(f"{t:<10.2f} | {t_precision:<10.4f} | {t_recall:<10.4f} | {t_f1:<10.4f}")

    # Print class-wise performance for 0.5 threshold
    print(f"\nBreakdown (Threshold 0.5):")
    print(f"True Negatives:  {conf_matrix[0][0]}")
    print(f"False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}")
    print(f"True Positives:  {conf_matrix[1][1]}")

if __name__ == "__main__":
    main()
