import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class LocallyConnected2D(layers.Layer):
    def __init__(self, filters, kernel_size, padding='valid', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding.lower()
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        _, h, w, c = input_shape
        kh, kw = self.kernel_size
        if self.padding == 'same':
            out_h, out_w = h, w
        else:
            out_h, out_w = h - kh + 1, w - kw + 1
        self.out_h, self.out_w = out_h, out_w
        self.kh, self.kw, self.in_c = kh, kw, c
        n_pos = out_h * out_w
        self.kernel = self.add_weight(
            name='kernel', shape=(n_pos, kh * kw * c, self.filters),
            initializer='glorot_uniform', trainable=True
        )
        self.bias = self.add_weight(
            name='bias', shape=(n_pos, self.filters),
            initializer='zeros', trainable=True
        )

    def call(self, inputs):
        kh, kw = self.kh, self.kw
        if self.padding == 'same':
            ph, pw = kh // 2, kw // 2
            inputs = tf.pad(inputs, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, kh, kw, 1], strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        batch = tf.shape(inputs)[0]
        patches = tf.reshape(patches, [batch, self.out_h * self.out_w, kh * kw * self.in_c])
        output = tf.einsum('bpi,pif->bpf', patches, self.kernel) + self.bias
        output = tf.reshape(output, [batch, self.out_h, self.out_w, self.filters])
        return self.activation(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activation': keras.activations.serialize(self.activation),
        })
        return config


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
        model.add(LocallyConnected2D(
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