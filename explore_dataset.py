import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image

def explore_dataset(data_dir):
    """
    Explore the dataset and generate visualizations
    """
    # Get class distribution
    class_counts = defaultdict(int)
    class_images = {}
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get class distribution and sample images
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Count images in class
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images
        
        # Get sample image
        if num_images > 0:
            img_name = os.listdir(class_dir)[0]
            img_path = os.path.join(class_dir, img_name)
            class_images[class_name] = img_path
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logs/class_distribution.png')
    plt.close()
    
    # Display sample images
    plt.figure(figsize=(15, 5))
    for i, (class_name, img_path) in enumerate(class_images.items(), 1):
        try:
            img = Image.open(img_path)
            plt.subplot(1, len(class_images), i)
            plt.imshow(img)
            plt.title(f"{class_name} (n={class_counts[class_name]})")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig('logs/sample_images.png')
    plt.close()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total classes: {len(class_counts)}")
    print(f"Total images: {sum(class_counts.values())}")
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count} images")
    
    return class_counts

if __name__ == "__main__":
    # Example usage
    train_dir = "Data/Train"
    test_dir = "Data/Test"
    
    print("Exploring training set...")
    train_stats = explore_dataset(train_dir)
    
    print("\nExploring test set...")
    test_stats = explore_dataset(test_dir)
