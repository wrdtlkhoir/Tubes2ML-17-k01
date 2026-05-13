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
            self.train_dir, seed=123, image_size=img_size, batch_size=batch_size
        )
        val_ds = keras.utils.image_dataset_from_directory(
            self.val_dir, seed=123, image_size=img_size, batch_size=batch_size
        )
        test_ds = keras.utils.image_dataset_from_directory(
            self.test_dir, seed=123, image_size=img_size, batch_size=batch_size
        )

        self.class_names = train_ds.class_names
        self.num_classes = len(self.class_names)

        train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
        val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
        test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

        return train_ds, val_ds, test_ds

    def train_model(self, model, train_ds, val_ds, model_name, epochs=20):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
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
        y_true, y_pred = [], []

        for images, labels in test_ds:
            predictions = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        return macro_f1, y_true, y_pred

    def _serialize_history(self, history):
        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def train_conv2d_variations(self, train_ds, val_ds, test_ds):
        # Full factorial: 2 x 2 x 2 x 2 = 16 architectures
        num_layers_options = [2, 3]
        filters_options = [
            (32, 64, 128),   # F1
            (64, 128, 256),  # F2
        ]
        kernel_options = [
            (3, 3, 3),  # K1
            (5, 5, 5),  # K2
        ]
        pooling_options = ['max', 'avg']

        conv2d_results = {}

        for num_layers in num_layers_options:
            for f_idx, filters in enumerate(filters_options, start=1):
                for k_idx, kernels in enumerate(kernel_options, start=1):
                    for pool_type in pooling_options:
                        model_name = f'conv2d_L{num_layers}_F{f_idx}_K{k_idx}_P{pool_type}'
                        print(f"Training {model_name}...")

                        model = build_conv2d_model(
                            num_conv_layers=num_layers,
                            filters_list=filters[:num_layers],
                            kernel_sizes_list=kernels[:num_layers],
                            pooling_type=pool_type,
                            num_classes=self.num_classes
                        )

                        history = self.train_model(model, train_ds, val_ds, model_name)
                        macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)

                        model_path = os.path.join(self.output_dir, f'{model_name}.h5')
                        model.save(model_path)

                        conv2d_results[model_name] = {
                            'macro_f1': float(macro_f1),
                            'num_layers': num_layers,
                            'filters': list(filters[:num_layers]),
                            'kernel_sizes': list(kernels[:num_layers]),
                            'pooling_type': pool_type,
                            'model_path': model_path,
                            'history': self._serialize_history(history),
                        }

                        print(f"  Macro F1-Score: {macro_f1:.4f}")

        self.results['conv2d'] = conv2d_results

    def get_best_conv2d_model(self):
        conv2d_results = self.results.get('conv2d', {})
        if not conv2d_results:
            raise ValueError("No Conv2D results found. Run train_conv2d_variations first.")
        best_name = max(conv2d_results, key=lambda k: conv2d_results[k]['macro_f1'])
        return best_name, conv2d_results[best_name]

    def train_locally_connected(self, train_ds, val_ds, test_ds, num_layers, filters, kernels, pool_type):
        model_name = 'locally_connected2d_best'
        print(f"Training {model_name}...")

        model = build_locally_connected_model(
            num_lc_layers=num_layers,
            filters_list=filters,
            kernel_sizes_list=kernels,
            pooling_type=pool_type,
            num_classes=self.num_classes
        )

        history = self.train_model(model, train_ds, val_ds, model_name)
        macro_f1, y_true, y_pred = self.evaluate_model(model, test_ds, model_name)

        model_path = os.path.join(self.output_dir, f'{model_name}.h5')
        model.save(model_path)

        self.results['locally_connected2d'] = {
            model_name: {
                'macro_f1': float(macro_f1),
                'num_layers': num_layers,
                'filters': list(filters),
                'kernel_sizes': list(kernels),
                'pooling_type': pool_type,
                'model_path': model_path,
                'history': self._serialize_history(history),
            }
        }

        print(f"  Macro F1-Score: {macro_f1:.4f}")

    def save_results(self):
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    def run_all_training(self, train_ds, val_ds, test_ds):
        print("Starting Conv2D training with variations (16 architectures)...")
        self.train_conv2d_variations(train_ds, val_ds, test_ds)

        best_name, best_config = self.get_best_conv2d_model()
        print(f"\nBest Conv2D model: {best_name} (F1={best_config['macro_f1']:.4f})")

        print("\nStarting LocallyConnected2D training with best architecture...")
        self.train_locally_connected(
            train_ds, val_ds, test_ds,
            num_layers=best_config['num_layers'],
            filters=best_config['filters'],
            kernels=best_config['kernel_sizes'],
            pool_type=best_config['pooling_type'],
        )

        self.save_results()
