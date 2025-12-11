import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

class RipenessDetector:
    def __init__(self, num_classes=3, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = self.build_model()
        self.class_names = ['Overripe', 'Ripe', 'Unripe']
        
    def build_model(self):
        """Build the CNN model with transfer learning"""
        # Load pre-trained EfficientNetB0
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Build the model
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, test_generator, epochs=20, batch_size=32):
        """Train the model"""
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                'models/ripeness_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Save the final model
        self.model.save('models/ripeness_model_final.h5')
        
        return history
    
    def evaluate(self, test_generator, history=None):
        """Evaluate the model and generate metrics"""
        # Evaluate the model
        y_pred = self.model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        
        # Plot training history if available
        if history is not None:
            self.plot_training_history(history)
        
        return report, cm
    
    def plot_training_history(self, history):
        """Plot training history"""
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
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.close()
    
    def save_model(self, path='models/ripeness_detector.joblib'):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    
    @classmethod
    def load_model(cls, path='models/ripeness_detector.joblib'):
        """Load the model from disk"""
        return joblib.load(path)
