import os
import shutil
import random

def split_dataset():
    # Configuration
    src_dir = 'dataset_original'
    dest_dir = 'dataset'
    classes = ['Normal', 'Tuberculosis']
    split_ratios = {'train': 0.75, 'val': 0.15, 'test': 0.10}
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Ensure destination directories exist
    if os.path.exists(dest_dir):
        print(f"Cleaning up existing {dest_dir} directory...")
        shutil.rmtree(dest_dir)

    for split in split_ratios.keys():
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    print(f"{'Split':<10} | {'Normal':<10} | {'Tuberculosis':<15}")
    print("-" * 40)

    counts = {split: {cls: 0 for cls in classes} for split in split_ratios.keys()}

    for cls in classes:
        cls_src_path = os.path.join(src_dir, cls)
        if not os.path.exists(cls_src_path):
            print(f"Warning: Source directory {cls_src_path} not found.")
            continue

        # Get all valid image files
        images = [f for f in os.listdir(cls_src_path) if f.lower().endswith(valid_extensions)]
        
        # Shuffle images
        random.shuffle(images)

        total_images = len(images)
        train_end = int(total_images * split_ratios['train'])
        val_end = train_end + int(total_images * split_ratios['val'])

        # Split images
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Copy files
        for split_name, split_images in splits.items():
            for img in split_images:
                src_img_path = os.path.join(cls_src_path, img)
                dest_img_path = os.path.join(dest_dir, split_name, cls, img)
                shutil.copy2(src_img_path, dest_img_path)
                counts[split_name][cls] += 1

    # Print results
    for split in split_ratios.keys():
        print(f"{split.capitalize():<10} | {counts[split]['Normal']:<10} | {counts[split]['Tuberculosis']:<15}")

if __name__ == "__main__":
    split_dataset()
