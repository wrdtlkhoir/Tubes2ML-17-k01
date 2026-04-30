import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_conv2d_model(input_shape=(224, 224, 3), num_classes=6, num_conv_layers=3, filters_list=(32, 64, 128), kernel_sizes_list=(3, 3, 3), pooling_type='max'):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_conv_layers):
        model.add(layers.Conv2D(
            filters=filters_list[i],
            kernel_size=kernel_sizes_list[i],
            padding='same',
            activation='relu'
        ))
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_locally_connected_model(input_shape=(224, 224, 3),num_classes=6, num_lc_layers=3, filters_list=(32, 64, 128), kernel_sizes_list=(3, 3, 3), pooling_type='max'):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(num_lc_layers):
        model.add(layers.LocallyConnected2D(
            filters=filters_list[i],
            kernel_size=kernel_sizes_list[i],
            padding='valid',
            activation='relu'
        ))
        
        if pooling_type == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model