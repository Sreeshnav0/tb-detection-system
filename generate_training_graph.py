import matplotlib.pyplot as plt
import os

def generate_graph():
    # 1. Accuracy values provided by user
    train_accuracy = [0.70, 0.78, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98]
    validation_accuracy = [0.68, 0.75, 0.82, 0.87, 0.90, 0.92, 0.94, 0.95, 0.955, 0.96]
    
    # Epochs range from 1 to 10
    epochs = range(1, len(train_accuracy) + 1)
    
    # 2. Create the graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, 'ro-', label='Validation Accuracy')
    
    # Formatting
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 3. Create outputs/ folder if it does not exist
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # 4. Save the graph
    output_path = os.path.join(output_dir, 'training_validation_accuracy.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Graph successfully saved at: {output_path}")

if __name__ == '__main__':
    generate_graph()
