"""
Baseline model architecture and training for SUQ-3 pipeline
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import config

class BaselineModel:
    """Build and train baseline GRU model"""
    
    def __init__(self, input_dim=config.INPUT_DIM, n_classes=config.N_CLASSES):
        """
        Initialize baseline model architecture
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model = None
        
    def build(self):
        """Build baseline GRU architecture"""
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Reshape((self.input_dim, 1))(inputs)
        x = layers.GRU(config.GRU_UNITS)(x)
        x = layers.Dense(config.DENSE_1_UNITS, activation='relu')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Baseline model built:")
        self.model.summary()
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=config.BASE_EPOCHS):
        """
        Train baseline model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            
        Returns:
            History object
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=15, 
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        return history
    
    def save(self, path):
        """Save model to disk"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
