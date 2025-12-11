import os
import json
import numpy as np
from data_preprocessing import create_data_generators
from model import RipenessDetector
from datetime import datetime

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Data paths
    train_dir = 'Data/Train'
    test_dir = 'Data/Test'
    
    # Hyperparameters
    batch_size = 32
    img_size = (224, 224)
    epochs = 30
    
    print("Creating data generators...")
    train_generator, test_generator = create_data_generators(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        img_size=img_size
    )
    
    # Get class names from the generator
    class_names = list(train_generator.class_indices.keys())
    print(f"Detected classes: {class_names}")
    
    # Initialize and train the model
    print("\nInitializing model...")
    detector = RipenessDetector(num_classes=len(class_names), img_size=img_size[0])
    detector.class_names = class_names  # Update class names
    
    print("\nStarting training...")
    history = detector.train(
        train_generator=train_generator,
        test_generator=test_generator,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    report, cm = detector.evaluate(test_generator, history=history)
    
    # Save evaluation results
    with open('logs/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save the model
    print("\nSaving model...")
    detector.save_model()
    
    print("\nTraining completed successfully!")
    print(f"Model and logs saved in 'models/' and 'logs/' directories.")

if __name__ == "__main__":
    main()
