import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

def load_data(data_dir, img_size=(224, 224)):
    """
    Load and preprocess images from the given directory
    """
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(X), np.array(y), class_names

def create_data_generators(train_dir, test_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, test_generator

def get_class_weights(y):
    """
    Calculate class weights to handle class imbalance
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    class_weights = compute_class_weight('balanced', 
                                       classes=np.unique(y), 
                                       y=y)
    return dict(enumerate(class_weights))
