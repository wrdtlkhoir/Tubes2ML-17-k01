import os
import numpy as np
import json
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from .models import build_conv2d_model, build_locally_connected_model


class ModelTrainer:
    def __init__(self, train_dir, val_dir, test_dir, output_dir='./trained_models'):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, img_size=(224, 224), batch_size=32):
        train_ds = keras.utils.image_dataset_from_directory(
            self.train_dir,
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            self.val_dir,
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )
        
        test_ds = keras.utils.image_dataset_from_directory(
            self.test_dir,
            seed=123,
            image_size=img_size,
            batch_size=batch_size
        )
        
        train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
        val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
        test_ds = test_ds.map(lambda x, y: (x / 255.0, y))
        
        self.num_classes = len(train_ds.class_names)
        self.class_names = train_ds.class_names
        
        return train_ds, val_ds, test_ds
    
    def train_model(self, model, train_ds, val_ds, model_name, epochs=20):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return history
    
    def evaluate_model(self, model, test_ds, model_name):
        y_true = []
        y_pred = []
        
        for images, labels in test_ds:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        return macro_f1, y_true, y_pred
    
    def train_conv2d_variations(self, train_ds, val_ds, test_ds):
        conv2d_results = {}
        
        num_conv_layers_variations = [2, 3, 4]
        filters_variations = [
            (32, 64, 128),
            (64, 128, 256),
            (16, 32, 64)
        ]
        kernel_sizes_variations = [
            (3, 3, 3),
            (5, 5, 5),
            (3, 5, 3)
        ]
        pooling_types = ['max', 'avg']
        
        var_idx = 0
        
        for num_layers in num_conv_layers_variations:
            filters = filters_variations[0][:num_layers]
            kernels = kernel_sizes_variations[0][:num_layers]
            
            for pool_type in pooling_types:
                var_idx += 1
                model_name = f'conv2d_layers_{num_layers}_pool_{pool_type}'
                print(f"Training {model_name}...")
                
                model = build_conv2d_model(
                    num_conv_layers=num_layers,
                    filters_list=filters,
                    kernel_sizes_list=kernels,
                    pooling_type=pool_type
                )
                
                history = self.train_model(model, train_ds, val_ds, model_name)
                macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)
                
                model_path = os.path.join(self.output_dir, f'{model_name}.h5')
                model.save(model_path)
                
                conv2d_results[model_name] = {
                    'macro_f1': float(macro_f1),
                    'num_layers': num_layers,
                    'pooling_type': pool_type,
                    'model_path': model_path
                }
                
                print(f"  Macro F1-Score: {macro_f1:.4f}")
        
        for i, filters in enumerate(filters_variations):
            model_name = f'conv2d_filters_{i+1}'
            print(f"Training {model_name}...")
            
            model = build_conv2d_model(
                num_conv_layers=3,
                filters_list=filters,
                kernel_sizes_list=kernel_sizes_variations[0][:3],
                pooling_type='max'
            )
            
            history = self.train_model(model, train_ds, val_ds, model_name)
            macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)
            
            model_path = os.path.join(self.output_dir, f'{model_name}.h5')
            model.save(model_path)
            
            conv2d_results[model_name] = {
                'macro_f1': float(macro_f1),
                'filters': filters,
                'model_path': model_path
            }
            
            print(f"  Macro F1-Score: {macro_f1:.4f}")
        
        for i, kernels in enumerate(kernel_sizes_variations):
            model_name = f'conv2d_kernels_{i+1}'
            print(f"Training {model_name}...")
            
            model = build_conv2d_model(
                num_conv_layers=3,
                filters_list=filters_variations[0],
                kernel_sizes_list=kernels,
                pooling_type='max'
            )
            
            history = self.train_model(model, train_ds, val_ds, model_name)
            macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)
            
            model_path = os.path.join(self.output_dir, f'{model_name}.h5')
            model.save(model_path)
            
            conv2d_results[model_name] = {
                'macro_f1': float(macro_f1),
                'kernel_sizes': kernels,
                'model_path': model_path
            }
            
            print(f"  Macro F1-Score: {macro_f1:.4f}")
        
        self.results['conv2d'] = conv2d_results
    
    def train_locally_connected(self, train_ds, val_ds, test_ds):
        model_name = 'locally_connected2d_baseline'
        print(f"Training {model_name}...")
        
        model = build_locally_connected_model(
            num_lc_layers=3,
            filters_list=(32, 64, 128),
            kernel_sizes_list=(3, 3, 3),
            pooling_type='max'
        )
        
        history = self.train_model(model, train_ds, val_ds, model_name)
        macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)
        
        model_path = os.path.join(self.output_dir, f'{model_name}.h5')
        model.save(model_path)
        
        self.results['locally_connected2d'] = {
            model_name: {
                'macro_f1': float(macro_f1),
                'model_path': model_path
            }
        }
        
        print(f"  Macro F1-Score: {macro_f1:.4f}")
    
    def save_results(self):
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    def run_all_training(self, train_ds, val_ds, test_ds):
        print("Starting Conv2D training with variations...")
        self.train_conv2d_variations(train_ds, val_ds, test_ds)
        
        print("\nStarting LocallyConnected2D training...")
        self.train_locally_connected(train_ds, val_ds, test_ds)
        
        self.save_results()
