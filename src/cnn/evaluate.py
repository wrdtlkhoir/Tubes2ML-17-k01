import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score, classification_report
from .layers import Conv2D, LocallyConnected2D, Dense
from .forward_propagation import ForwardPropagation
import os


class ModelEvaluator:
    def __init__(self, trained_models_dir):
        self.trained_models_dir = trained_models_dir
    
    def get_model_parameter_count(self, model):
        return model.count_params()
    
    def inference_keras(self, model, test_ds):
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_ds:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
            y_pred_proba.extend(predictions)
        
        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)
    
    def inference_from_scratch(self, fp_model, test_ds):
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_ds:
            images_np = images.numpy()
            predictions = fp_model.forward(images_np)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
            y_pred_proba.extend(predictions)
        
        return np.array(y_true), np.array(y_pred), np.array(y_pred_proba)
    
    def calculate_metrics(self, y_true, y_pred):
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = np.mean(y_true == y_pred)
        
        return {
            'macro_f1': macro_f1,
            'accuracy': accuracy
        }
    
    def load_keras_model_weights_to_fp(self, keras_model, layer_names):
        fp = ForwardPropagation()
        
        for layer_name in layer_names:
            keras_layer = keras_model.get_layer(layer_name)
            layer_type = type(keras_layer).__name__
            
            if layer_type == 'Conv2D':
                kernel, bias = keras_layer.get_weights()
                fp.add_conv2d(kernel, bias, stride=keras_layer.strides[0], padding=0)
            
            elif layer_type == 'LocallyConnected2D':
                kernel, bias = keras_layer.get_weights()
                fp.add_locally_connected2d(kernel, bias, stride=keras_layer.strides[0], padding=0)
            
            elif layer_type == 'MaxPooling2D':
                fp.add_maxpooling2d(pool_size=keras_layer.pool_size, strides=keras_layer.strides)
            
            elif layer_type == 'AveragePooling2D':
                fp.add_avgpooling2d(pool_size=keras_layer.pool_size, strides=keras_layer.strides)
            
            elif layer_type == 'GlobalAveragePooling2D':
                fp.add_global_avgpooling2d()
            
            elif layer_type == 'GlobalMaxPooling2D':
                fp.add_global_maxpooling2d()
            
            elif layer_type == 'Flatten':
                fp.add_flatten()
            
            elif layer_type == 'Dense':
                weight, bias = keras_layer.get_weights()
                activation = keras_layer.activation.__name__ if hasattr(keras_layer, 'activation') else None
                fp.add_dense(weight, bias, activation)
        
        return fp
    
    def evaluate_model_pair(self, model_name, test_ds, layer_names=None):
        model_path = os.path.join(self.trained_models_dir, f'{model_name}.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        keras_model = keras.models.load_model(model_path)
        
        y_true_keras, y_pred_keras, y_pred_proba_keras = self.inference_keras(keras_model, test_ds)
        metrics_keras = self.calculate_metrics(y_true_keras, y_pred_keras)
        
        if layer_names is None:
            layer_names = [l.name for l in keras_model.layers]
        
        fp_model = self.load_keras_model_weights_to_fp(keras_model, layer_names)
        y_true_fs, y_pred_fs, y_pred_proba_fs = self.inference_from_scratch(fp_model, test_ds)
        metrics_fs = self.calculate_metrics(y_true_fs, y_pred_fs)
        
        param_count = self.get_model_parameter_count(keras_model)
        
        return {
            'model_name': model_name,
            'keras_metrics': metrics_keras,
            'from_scratch_metrics': metrics_fs,
            'parameter_count': param_count,
            'y_true': y_true_keras,
            'y_pred_keras': y_pred_keras,
            'y_pred_from_scratch': y_pred_fs,
            'predictions_match': np.allclose(y_pred_proba_keras, y_pred_proba_fs, atol=1e-5)
        }
    
    def evaluate_all_models(self, model_names, test_ds):
        results = {}
        
        for model_name in model_names:
            print(f"Evaluating {model_name}...")
            try:
                result = self.evaluate_model_pair(model_name, test_ds)
                results[model_name] = result
                print(f"  Keras F1: {result['keras_metrics']['macro_f1']:.4f}")
                print(f"  From Scratch F1: {result['from_scratch_metrics']['macro_f1']:.4f}")
                print(f"  Predictions match: {result['predictions_match']}")
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def compare_shared_vs_non_shared(self, test_ds):
        conv2d_model_path = os.path.join(self.trained_models_dir, 'conv2d_layers_3_pool_max.h5')
        lc_model_path = os.path.join(self.trained_models_dir, 'locally_connected2d_baseline.h5')
        
        results = {}
        
        if os.path.exists(conv2d_model_path):
            conv2d_result = self.evaluate_model_pair('conv2d_layers_3_pool_max', test_ds)
            results['conv2d'] = conv2d_result
            print(f"Conv2D (shared): F1={conv2d_result['keras_metrics']['macro_f1']:.4f}, "
                  f"Params={conv2d_result['parameter_count']}")
        
        if os.path.exists(lc_model_path):
            lc_result = self.evaluate_model_pair('locally_connected2d_baseline', test_ds)
            results['locally_connected2d'] = lc_result
            print(f"LocallyConnected2D (non-shared): F1={lc_result['keras_metrics']['macro_f1']:.4f}, "
                  f"Params={lc_result['parameter_count']}")
        
        return results
