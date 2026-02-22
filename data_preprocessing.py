import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

class DataPreprocessor:
    def __init__(self, data_dir='Data', img_size=(224, 224), batch_size=32, test_size=0.2, val_size=0.2):
        """
        Initialize the DataPreprocessor
        
        Args:
            data_dir (str): Root directory containing Train and Test folders
            img_size (tuple): Target image dimensions (height, width)
            batch_size (int): Batch size for data generators
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.class_indices = None
        self.class_names = None
        
        # Data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=val_size
        )
        
        # No augmentation for validation and test
        self.test_datagen = ImageDataGenerator()
    
    def get_data_generators(self):
        """
        Create and return data generators for training, validation, and testing
        
        Returns:
            tuple: (train_generator, val_generator, test_generator, class_weights)
        """
        # Create training generator
        train_generator = self.train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'Train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Store class indices and names
        self.class_indices = train_generator.class_indices
        self.class_names = list(self.class_indices.keys())
        
        # Save class indices
        self._save_class_indices(train_generator)
        
        # Create validation generator
        val_generator = self.train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'Train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Create test generator
        test_generator = self.test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'Test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Calculate class weights for imbalanced data
        class_weights = self._calculate_class_weights(train_generator)
        
        return train_generator, val_generator, test_generator, class_weights
    
    def _calculate_class_weights(self, generator):
        """Calculate class weights to handle imbalanced data"""
        class_counts = np.bincount(generator.classes)
        total_samples = len(generator.classes)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            class_weights[i] = total_samples / (len(class_counts) * count)
            
        return class_weights
    
    def _save_class_indices(self, generator):
        """Save class indices to a JSON file for later use in predictions"""
        if hasattr(generator, 'class_indices'):
            class_indices = {v: k for k, v in generator.class_indices.items()}
            os.makedirs('models', exist_ok=True)
            with open('models/class_indices.json', 'w') as f:
                json.dump(class_indices, f, indent=4)
    
    def plot_sample_images(self, generator, num_images=5):
        """Plot sample images from the generator"""
        x_batch, y_batch = next(generator)
        
        # Get class names from the generator
        class_indices = generator.class_indices
        class_names = list(class_indices.keys())
        plt.figure(figsize=(15, 5))
        for i in range(min(num_images, len(x_batch))):
            plt.subplot(1, num_images, i+1)
            plt.imshow(x_batch[i].astype('uint8'))
            plt.title(f"{class_names[np.argmax(y_batch[i])]}")
            plt.axis('off')
        plt.tight_layout()
        os.makedirs('logs', exist_ok=True)
        plt.savefig('logs/sample_images.png')
        plt.close()
        
    def get_class_distribution(self, data_dir=None):
        """
        Calculate the number of images in each class
        
        Args:
            data_dir (str): Directory containing class subdirectories. If None, uses the train directory.
            
        Returns:
            dict: Dictionary with class names as keys and image counts as values
        """
        if data_dir is None:
            data_dir = os.path.join(self.data_dir, 'Train')
            
        class_counts = {}
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        
        # Count images in each class
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                num_images = len([f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = num_images
        
        return class_counts
    
    def plot_class_distribution(self, class_counts=None, save_path=None):
        """
        Plot the class distribution
        
        Args:
            class_counts (dict): Dictionary with class names as keys and counts as values
            save_path (str): Path to save the plot. If None, just shows the plot.
        """
        if class_counts is None:
            class_counts = self.get_class_distribution()
            
        if not class_counts:
            print("No classes found to plot.")
            return
        
        # Sort classes by count for better visualization
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        class_names = [c[0].replace('_', ' ').title() for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, counts, color='skyblue')
        
        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title('Class Distribution in Dataset', fontsize=14, pad=20)
        plt.xlabel('Class', fontsize=12, labelpad=10)
        plt.ylabel('Number of Images', fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
        return class_counts

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Analyze and plot class distribution
    print("Analyzing dataset...")
    class_counts = preprocessor.plot_class_distribution(
        save_path=os.path.join('logs', 'class_distribution.png')
    )
    
    if class_counts:
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{class_name}: {count} images")
    
    # Get data generators
    print("\nPreparing data generators...")
    train_gen, val_gen, test_gen, class_weights = preprocessor.get_data_generators()
    
    # Print class indices and weights
    print("\nClass indices:", preprocessor.class_indices)
    print("Class weights:", class_weights)
    
    # Get one batch of data
    x_batch, y_batch = next(train_gen)
    print(f"\nBatch shape: {x_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Number of classes: {y_batch.shape[1]}")
    
    # Plot sample images
    preprocessor.plot_sample_images(train_gen)
    print("Sample images saved to logs/sample_images.png")
