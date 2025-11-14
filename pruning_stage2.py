"""
Stage 2: Unstructured Magnitude Pruning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
import numpy as np
import math
import config

class UnstructuredPruning:
    """Stage 2: Unstructured magnitude pruning"""
    
    def __init__(self, stage1_model, X_train, batch_size=config.BATCH_SIZE):
        """
        Initialize unstructured pruning
        
        Args:
            stage1_model: Pruned model from Stage 1
            X_train: Training data for schedule calculation
            batch_size: Batch size for training
        """
        self.stage1_model = stage1_model
        self.X_train = X_train
        self.batch_size = batch_size
        self.model = None
        self.pruned_model = None
        
    def _create_pruning_schedule(self):
        """Create polynomial decay pruning schedule"""
        steps_per_epoch = math.ceil(len(self.X_train) / self.batch_size)
        end_step = steps_per_epoch * config.PRUNE_UNSTR_FT_EPOCHS
        
        schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=config.UNSTRUCTURED_PRUNING_CONFIG['initial_sparsity'],
            final_sparsity=config.UNSTRUCTURED_PRUNING_CONFIG['final_sparsity'],
            begin_step=config.UNSTRUCTURED_PRUNING_CONFIG['begin_step'],
            end_step=end_step
        )
        return schedule
    
    def build_pruned_model(self):
        """Build model with unstructured pruning wrappers"""
        schedule = self._create_pruning_schedule()
        
        pruning_params = {
            'pruning_schedule': schedule,
        }
        
        # Build model structure (same as baseline but with pruning)
        inputs = keras.Input(shape=(config.INPUT_DIM,))
        x = layers.Reshape((config.INPUT_DIM, 1))(inputs)
        x = layers.GRU(config.GRU_UNITS)(x)
        
        # Wrap Dense layers with magnitude pruning
        dense_1 = layers.Dense(config.DENSE_1_UNITS, activation='relu', name='dense_1')
        dense_1_wrapped = tfmot.sparsity.keras.PruneLowMagnitude(dense_1, **pruning_params)
        x = dense_1_wrapped(x)
        
        dense_2 = layers.Dense(config.N_CLASSES, activation='softmax', name='dense_2')
        dense_2_wrapped = tfmot.sparsity.keras.PruneLowMagnitude(dense_2, **pruning_params)
        outputs = dense_2_wrapped(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Transfer weights from Stage 1 pruned model
        self.model.layers[1].set_weights(self.stage1_model.layers[1].get_weights())  # Reshape
        self.model.layers[2].set_weights(self.stage1_model.layers[2].get_weights())  # GRU
        self.model.layers[3].layer.set_weights(self.stage1_model.layers[3].get_weights())  # Dense 1
        self.model.layers[4].layer.set_weights(self.stage1_model.layers[4].get_weights())  # Dense 2
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Unstructured pruning model built")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=config.PRUNE_UNSTR_FT_EPOCHS):
        """Train with unstructured pruning"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[
                tfmot.sparsity.keras.UpdatePruningStep(),
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            ],
            verbose=1
        )
        return history
    
    def strip_pruning(self):
        """Strip pruning wrappers to create final model"""
        self.pruned_model = tfmot.sparsity.keras.strip_pruning(self.model)
        self.pruned_model.compile(
            optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Pruning wrappers stripped")
        return self.pruned_model
    
    def save(self, path):
        """Save pruned model"""
        self.pruned_model.save(path)
        print(f"Unstructured pruned model saved to {path}")
