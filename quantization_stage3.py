"""
Stage 3: Quantization-Aware Training and TFLite Export
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
import numpy as np
import time
import config

class QuantizationAwareTraining:
    """Stage 3: QAT and TFLite export"""
    
    def __init__(self, stage2_model, X_train):
        """
        Initialize QAT
        
        Args:
            stage2_model: Pruned model from Stage 2
            X_train: Training data for representative dataset
        """
        self.stage2_model = stage2_model
        self.X_train = X_train
        self.qat_model = None
        self.tflite_model = None
        
    def prepare_representative_dataset(self, num_samples=200):
        """Create representative dataset for quantization calibration"""
        def representative_gen():
            for i in range(min(num_samples, len(self.X_train))):
                yield [self.X_train[i:i+1].astype(np.float32)]
        return representative_gen
    
    def apply_qat(self):
        """Apply Quantization-Aware Training"""
        try:
            self.qat_model = tfmot.quantization.keras.quantize_model(self.stage2_model)
            self.qat_model.compile(
                optimizer=keras.optimizers.Adam(config.LEARNING_RATE_QAT),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("QAT model created successfully")
            return self.qat_model
        except RuntimeError as e:
            print(f"QAT Error (likely GRU unsupported): {e}")
            print("Using Stage 2 model as fallback for TFLite export")
            self.qat_model = self.stage2_model
            return self.qat_model
    
    def train_qat(self, X_train, y_train, X_val, y_val, epochs=config.QAT_EPOCHS):
        """Train with QAT"""
        if self.qat_model is None:
            self.apply_qat()
        
        history = self.qat_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )
        return history
    
    def convert_to_tflite(self, output_path='models/suq3_int8.tflite'):
        """Convert model to TFLite INT8 format"""
        representative_gen = self.prepare_representative_dataset()
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.qat_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        self.tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(self.tflite_model)
        
        print(f"TFLite INT8 model saved to {output_path}")
        return output_path
    
    def benchmark_tflite(self, tflite_path, X_test, num_runs=200):
        """Benchmark TFLite model latency"""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        
        sample = X_test[:1].astype(np.float32)
        
        # Quantize input if needed
        scale, zp = inp['quantization']
        if scale == 0:
            q_sample = sample.astype(np.float32)
        else:
            q_sample = (sample / scale + zp).astype(np.int8)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(inp['index'], q_sample)
            interpreter.invoke()
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            interpreter.set_tensor(inp['index'], q_sample)
            interpreter.invoke()
        end = time.time()
        
        avg_latency_ms = (end - start) / num_runs * 1000
        print(f"Average TFLite inference latency: {avg_latency_ms:.4f} ms")
        
        return avg_latency_ms
    
    def save_qat_model(self, path):
        """Save QAT Keras model"""
        self.qat_model.save(path)
        print(f"QAT model saved to {path}")
