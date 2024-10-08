"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""

import os
import sys
import argparse
import pdb
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Flatten
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.backend as backend

from tensorflow.keras import models
import tensorflow.keras.utils as keras_utils
import tensorflow.keras.backend as backend

# import imagenet_utils
# from imagenet_utils import get_submodules_from_kwargs
# from imagenet_utils import decode_predictions
# from imagenet_utils import _obtain_input_shape

# preprocess_input = imagenet_utils.preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block, r=1):
    filters1, filters2, filters3 = [int(f * r) for f in filters]
    if backend.image_data_format() == 'channels_last':
        bn_axis = 2
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters1, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, r=1, strides=2):
    filters1, filters2, filters3 = [int(f * r) for f in filters]
    if backend.image_data_format() == 'channels_last':
        bn_axis = 2
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters1, 1, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv1D(filters3, 1, strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def create_model(input_shape, classes,pruning_rate, pooling=None, include_top=True, **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 1D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    print(pruning_rate)
    if backend.image_data_format() == 'channels_last':
        bn_axis = 2
    else:
        bn_axis = 1
    print("DSADDSADASDDS---------------",input_shape)
    img_input = layers.Input(shape=input_shape)

    x = layers.ZeroPadding1D(padding=3, name='conv1_pad')(img_input)
    x = layers.Conv1D(64, 7,
                      strides=2,
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding1D(padding=1, name='pool1_pad')(x)
    x = layers.MaxPooling1D(3, strides=2)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1,r=pruning_rate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',r=pruning_rate)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',r=pruning_rate)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',r=pruning_rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',r=pruning_rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',r=pruning_rate)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',r=pruning_rate)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',r=pruning_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',r=pruning_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',r=pruning_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',r=pruning_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',r=pruning_rate)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',r=pruning_rate)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',r=pruning_rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',r=pruning_rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',r=pruning_rate)
    classes=27
    x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    
  

    # Create model.
    inputs = img_input
    model = models.Model(inputs, x, name='resnet50')

    return model
