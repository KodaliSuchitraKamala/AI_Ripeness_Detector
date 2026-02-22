import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import os
import json

class RipenessModel:
    def __init__(self, num_classes=12, img_size=(224, 224, 3), learning_rate=1e-4):
        """
        Initialize the Ripeness Detection Model
        
        Args:
            num_classes (int): Number of output classes (default: 12 for Unripe, Ripe, Overripe)
            img_size (tuple): Input image dimensions (height, width, channels)
            learning_rate (float): Initial learning rate for the optimizer
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.total_epochs = 20  # Default value, will be updated by set_total_epochs
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the enhanced model architecture using EfficientNetB0 as base"""
        # Load pre-trained EfficientNetB0 without top layers
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.img_size,
            pooling=None
        )
        
        # Freeze the base model layers initially
        base_model.trainable = False
        
        # Create new model on top
        inputs = tf.keras.Input(shape=self.img_size)
        
        # Preprocess input for EfficientNet
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        
        # Enhanced model head
        x = GlobalAveragePooling2D()(x)
        
        # First dense layer with 1024 units
        x = Dense(1024, activation='swish', name='dense')(x)
        x = BatchNormalization(name='batch_normalization')(x)
        x = Dropout(0.3, name='dropout')(x)
        
        # Second dense layer with 512 units
        x = Dense(512, activation='swish', name='dense_1')(x)
        x = BatchNormalization(name='batch_normalization_1')(x)
        x = Dropout(0.2, name='dropout_1')(x)
        
        # SE block with matching dimensions
        se = Dense(32, activation='swish', name='dense_2')(x)
        se = Dense(512, activation='sigmoid', name='dense_3')(se)
        x = tf.keras.layers.Multiply(name='multiply')([x, se])
        
        # Output layer
        outputs = Dense(self.num_classes, 
                        activation='softmax',
                        name='dense_4',
                        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        
        # Create model
        model = Model(inputs, outputs)
        
        return model
    
    def compile_model(self):
        """Compile the model with optimizer, loss, and metrics"""
        # Use label smoothing for better generalization
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        # Use AdamW optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile with additional metrics
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
            ]
        )
    
    def get_callbacks(self, log_dir='logs/fit', checkpoint_path='models/ripeness_model.weights.h5'):
        """
        Get enhanced training callbacks with additional monitoring and logging
        
        Args:
            log_dir (str): Directory to save TensorBoard logs
            checkpoint_path (str): Path to save model checkpoints
            
        Returns:
            list: List of callbacks
        """
        # Create necessary directories
        os.makedirs(os.path.dirname(log_dir), exist_ok=True) 
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Enhanced Model checkpoint callback
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            save_freq='epoch',
            verbose=1
        )
        
        # Early stopping with more patience and monitoring multiple metrics
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=10,  # Increased patience
            verbose=1,
            mode='min',
            restore_best_weights=True,
            baseline=None
        )
        
        # Learning rate scheduler with warmup
        def lr_schedule(epoch, lr):
            """Learning rate schedule with warmup and cosine decay"""
            # Warmup for first 5 epochs
            if epoch < 5:
                return float(self.learning_rate * (epoch + 1) / 5)
            # Cosine decay after warmup
            else:
                return float(self.learning_rate * 0.5 * (1 + np.cos(epoch * np.pi / self.total_epochs)))
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        
        # ReduceLROnPlateau as a backup
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            min_delta=1e-4,
            verbose=1,
            mode='min'
        )
        
        # Enhanced TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='batch',
            profile_batch=0  # Disable profiling for better performance
        )
        
        # CSV Logger
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training.log'),
            append=True
        )
        
        return [
            checkpoint,
            early_stopping,
            lr_scheduler,
            reduce_lr,
            tensorboard,
            csv_logger
        ]
    
    def unfreeze_layers(self, num_layers=20):
        """
        Unfreeze the top N layers of the base model for fine-tuning
        
        Args:
            num_layers (int): Number of top layers to unfreeze
        """
        # Unfreeze the base model
        self.model.trainable = True
        
        # Freeze the bottom layers
        for layer in self.model.layers[1].layers[:-num_layers]:
            layer.trainable = False
        
        # Get current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"Current learning rate: {current_lr}")
            
        # Recompile the model with the new configuration
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"Unfroze the last {num_layers} layers of the base model.")
        print(f"Learning rate set to:{self.model.optimizer}")
    
    def save_model(self, model_dir='models'):
        """Save the model architecture and weights"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(os.path.join(model_dir, 'ripeness_model.json'), 'w') as json_file:
            json_file.write(model_json)
            
        # Save model weights
        self.model.save_weights(os.path.join(model_dir, 'ripeness_model.weights.h5'))
        
        print(f"Model saved in {model_dir}")
    
    def load_weights(self, weights_path):
        """Load model weights from file"""
        self.model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
        
    def set_total_epochs(self, total_epochs):
        """Set the total number of epochs for learning rate scheduling"""
        self.total_epochs = total_epochs
        print(f"Set total epochs for learning rate scheduling: {self.total_epochs}")

if __name__ == "__main__":
    # Example usage
    model = RipenessModel()
    model.compile_model()
    model.model.summary()
