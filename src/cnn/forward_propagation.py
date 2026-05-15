import numpy as np
from typing import List, Dict, Any
from .layers import (
    Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, ReLU, Softmax, Dense
)


class ForwardPropagation:
    def __init__(self):
        self.layers = []
    
    def add_conv2d(self, kernel: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0):
        self.layers.append(Conv2D(kernel, bias, stride, padding))
    
    def add_locally_connected2d(self, kernel: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0,
                                kH: int = None, kW: int = None):
        self.layers.append(LocallyConnected2D(kernel, bias, stride, padding, kH=kH, kW=kW))
    
    def add_maxpooling2d(self, pool_size: tuple = (2, 2), strides: tuple = None):
        self.layers.append(MaxPooling2D(pool_size, strides))
    
    def add_avgpooling2d(self, pool_size: tuple = (2, 2), strides: tuple = None):
        self.layers.append(AveragePooling2D(pool_size, strides))
    
    def add_global_avgpooling2d(self):
        self.layers.append(GlobalAveragePooling2D())
    
    def add_global_maxpooling2d(self):
        self.layers.append(GlobalMaxPooling2D())
    
    def add_flatten(self):
        self.layers.append(Flatten())
    
    def add_relu(self):
        self.layers.append(ReLU())
    
    def add_softmax(self):
        self.layers.append(Softmax())
    
    def add_dense(self, weight: np.ndarray, bias: np.ndarray, activation: str = None):
        self.layers.append(Dense(weight, bias, activation))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output


def load_keras_model_weights(keras_model, layer_configs: List[Dict[str, Any]]) -> ForwardPropagation:
    fp = ForwardPropagation()
    
    for i, layer_config in enumerate(layer_configs):
        layer_type = layer_config['type']
        keras_layer_name = layer_config.get('name')
        
        if layer_type == 'conv2d':
            layer = keras_model.get_layer(keras_layer_name)
            kernel, bias = layer.get_weights()
            padding_val = 0 if layer.padding == 'valid' else layer.kernel_size[0] // 2
            fp.add_conv2d(kernel, bias, stride=layer.strides[0], padding=padding_val)
        
        elif layer_type == 'locally_connected2d':
            layer = keras_model.get_layer(keras_layer_name)
            kernel, bias = layer.get_weights()
            fp.add_locally_connected2d(
                kernel, bias,
                stride=layer.strides[0],
                padding=0 if layer.padding == 'valid' else 1,
                kH=layer.kh, kW=layer.kw,
            )
        
        elif layer_type == 'maxpooling2d':
            pool_size = layer_config.get('pool_size', (2, 2))
            strides = layer_config.get('strides', None)
            fp.add_maxpooling2d(pool_size, strides)
        
        elif layer_type == 'avgpooling2d':
            pool_size = layer_config.get('pool_size', (2, 2))
            strides = layer_config.get('strides', None)
            fp.add_avgpooling2d(pool_size, strides)
        
        elif layer_type == 'global_avgpooling2d':
            fp.add_global_avgpooling2d()
        
        elif layer_type == 'global_maxpooling2d':
            fp.add_global_maxpooling2d()
        
        elif layer_type == 'flatten':
            fp.add_flatten()
        
        elif layer_type == 'relu':
            fp.add_relu()
        
        elif layer_type == 'softmax':
            fp.add_softmax()
        
        elif layer_type == 'dense':
            layer = keras_model.get_layer(keras_layer_name)
            weight, bias = layer.get_weights()
            activation = layer_config.get('activation', None)
            fp.add_dense(weight, bias, activation)
    
    return fp
