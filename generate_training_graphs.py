import matplotlib.pyplot as plt
import os

def generate_graphs():
    # 1. Data values provided by user
    train_accuracy = [0.70, 0.78, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98]
    val_accuracy   = [0.68, 0.75, 0.82, 0.87, 0.90, 0.92, 0.94, 0.95, 0.955, 0.96]

    train_loss = [0.90, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    val_loss   = [0.95, 0.75, 0.65, 0.55, 0.45, 0.40, 0.35, 0.30, 0.28, 0.25]
    
    # Epochs range from 1 to 10
    epochs = range(1, len(train_accuracy) + 1)
    
    # Create outputs/ folder if it does not exist
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Accuracy Graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    accuracy_path = os.path.join(output_dir, 'training_validation_accuracy.png')
    plt.savefig(accuracy_path)
    plt.close()
    print(f"Accuracy graph saved at: {accuracy_path}")

    # 3. Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    loss_path = os.path.join(output_dir, 'training_validation_loss.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss graph saved at: {loss_path}")

if __name__ == '__main__':
    generate_graphs()
