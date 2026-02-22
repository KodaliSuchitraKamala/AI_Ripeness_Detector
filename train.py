import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import our custom modules
from data_preprocessing import DataPreprocessor
from model import RipenessModel

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'initial_epochs': 10,  # Initial training with frozen base
    'fine_tune_epochs': 10,  # Fine-tuning with unfrozen layers
    'learning_rate': 1e-4,
    'fine_tune_learning_rate': 1e-5,
    'num_classes': 12,  # Will be set dynamically based on data
    'data_dir': 'Data',
    'model_dir': 'models',
    'logs_dir': 'logs',
    'test_size': 0.2,
    'val_size': 0.2
}

def plot_training_history(history, fine_tune_history=None):
    """Plot training and validation metrics"""
    # Combine histories if fine-tuning was done
    if fine_tune_history is not None:
        for key in history.history.keys():
            if key in fine_tune_history.history:  # Only combine if key exists in both
                history.history[key] += fine_tune_history.history[key]
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Save the plot
    plt.tight_layout()
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('logs/confusion_matrix.png')
    plt.close()

def evaluate_model(model, test_generator, class_names):
    """Evaluate the model and generate metrics"""
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = model.model.evaluate(test_generator, verbose=1)
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true, 
        y_pred_classes, 
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    with open('logs/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    return test_metrics

def main():
    # Create necessary directories
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['logs_dir'], exist_ok=True)
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        data_dir=CONFIG['data_dir'],
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )
    
    # Get data generators
    print("Loading data...")
    train_gen, val_gen, test_gen, class_weights = preprocessor.get_data_generators()
    
    # In the main function, replace the class_names line with:
    class_names = preprocessor.class_names
    if class_names is None:
        # Fallback to using the generator's class indices if for some reason class_names is not set
        class_names = list(train_gen.class_indices.keys())
    
    # Initialize model
    print("Initializing model...")
    num_classes = train_gen.num_classes
    model = RipenessModel(
        num_classes=num_classes,
        img_size=CONFIG['img_size'] + (3,),
        learning_rate=CONFIG['learning_rate']
    )
    model.compile_model()
    
    # Set total epochs for learning rate scheduling
    total_epochs = CONFIG['initial_epochs'] + CONFIG['fine_tune_epochs']
    model.set_total_epochs(total_epochs)
    
    # Get callbacks
    log_dir = os.path.join(CONFIG['logs_dir'], 'fit', datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = os.path.join(CONFIG['model_dir'], 'ripeness_model.weights.h5')
    callbacks = model.get_callbacks(log_dir=log_dir, checkpoint_path=checkpoint_path)
    
    # Train the model (initial training with frozen base)
    print("\nStarting initial training...")
    history = model.model.fit(
        train_gen,
        epochs=CONFIG['initial_epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Fine-tune the model (unfreeze top layers)
    print("\nStarting fine-tuning...")
    model.unfreeze_layers(num_layers=20)
    
    # Update learning rate for fine-tuning
    model.model.optimizer.learning_rate = float(CONFIG['fine_tune_learning_rate'])
    print(f"Set fine-tuning learning rate to: {float(CONFIG['fine_tune_learning_rate'])}")
    
    # Continue training with unfrozen layers
    fine_tune_history = model.model.fit(
        train_gen,
        epochs=CONFIG['initial_epochs'] + CONFIG['fine_tune_epochs'],
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Save the final model
    model.save_model(CONFIG['model_dir'])
    
    # Evaluate the model
    evaluate_model(model, test_gen, class_names)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to {CONFIG['model_dir']}")
    print(f"Training logs saved to {log_dir}")

if __name__ == "__main__":
    main()
