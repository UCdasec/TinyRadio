#! /usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import \
    Dense, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Input, Activation, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, BatchNormalization, LSTM, Flatten, ELU, AveragePooling1D, Permute
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from complexnn import ComplexConv1D, Modrelu, SpectralPooling1D, ComplexDense
from complexnn import utils
import sys
sys.path.append('ADA')

import af_model, af_classifier


def create_convnet(window_size=288, channels=2, output_size=10):
    inputs = Input(shape=(288, 2))

    conv = ComplexConv1D(
        8, 8, strides=1,
        activation='relu')(inputs)
    pool = AveragePooling1D(pool_size=4, strides=2)(conv)

    pool = Permute([2, 1])(pool)
    flattened = Flatten()(pool)

    dense = ComplexDense(2048, activation='relu')(flattened)
    predictions = ComplexDense(
        output_size,
        activation='sigmoid',
        bias_initializer=Constant(value=-5))(dense)
    predictions = utils.GetReal(predictions)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_DF(inp_shape, class_num = 5, emb_size = 64, classification = False):
    # -----------------Entry flow -----------------
    input_data = Input(shape=inp_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2')(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation('relu', name='block2_act1')(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2')(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1')(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2')(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1')(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2')(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)

    output = Flatten()(model)

    if classification:
        dense_layer = Dense(class_num, name='FeaturesVec', activation='softmax')(output)
    else:
        dense_layer = Dense(emb_size, name='FeaturesVec')(output)
    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def create_ComplexDF(inp_shape, class_num = 5, emb_size = 64, weight_decay = 1e-4, classification = False):
    convArgs = dict(use_bias=False,
                    kernel_regularizer=l2(weight_decay),
                    spectral_parametrization=False,
                    kernel_initializer='complex')
    # -----------------Entry flow -----------------
    input_data = Input(shape=inp_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = ComplexConv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', **convArgs)(input_data)
    model = Modrelu(name='block1_adv_act1')(model)
    model = ComplexConv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', **convArgs)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    # model = SpectralPooling1D()(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = ComplexConv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same',  **convArgs)(model)
    model = Modrelu(name='block2_act1')(model)
    model = ComplexConv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same',  **convArgs)(model)
    model = Modrelu(name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    # model = SpectralPooling1D()(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = ComplexConv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same',  **convArgs)(model)
    model = Modrelu(name='block3_act1')(model)
    model = ComplexConv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same',  **convArgs)(model)
    model = Modrelu(name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    # model = SpectralPooling1D()(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = ComplexConv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same',  **convArgs)(model)
    model = Modrelu(name='block4_act1')(model)
    model = ComplexConv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same',  **convArgs)(model)
    model = Modrelu(name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)
    # model = SpectralPooling1D()(model)
    # model = utils.GetAbs(name="Abs")(model)
    output = Flatten()(model)

    if classification:
        dense_layer = Dense(class_num, name='FeaturesVec', activation='softmax')(output)
    else:
        dense_layer = Dense(emb_size, name='FeaturesVec')(output)
    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def create_2DDF(inp_shape, class_num = 5, emb_size = 64, classification = False):
    # -----------------Entry flow -----------------
    input_data = Input(shape=inp_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', (8,8), (8,8), (8,8), (8,8)]
    conv_stride_size = ['None', (1,1), (1,1), (1,1), (1,1)]
    pool_stride_size = ['None', (4,4), (4,4), (4,4), (4,4)]
    pool_size = ['None', (8,8), (8,8), (8,8), (8,8)]

    model = Conv2D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv2D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2')(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling2D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv2D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation('relu', name='block2_act1')(model)
    model = Conv2D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2')(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling2D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv2D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1')(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv2D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2')(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling2D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv2D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1')(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv2D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2')(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling2D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)

    output = Flatten()(model)

    if classification:
        dense_layer = Dense(class_num, name='FeaturesVec', activation='softmax')(output)
    else:
        dense_layer = Dense(emb_size, name='FeaturesVec')(output)
    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def complex_baselineBlock(input, weight_decay, block_idx):
    convArgs = dict(use_bias=False,
                    kernel_regularizer=l2(weight_decay),
                    spectral_parametrization=False,
                    kernel_initializer='complex')

    kernel_size = ['None', 7, 5]
    filter_num = ['None', 128, 128]
    conv_stride = ['None', 1, 1]
    pool_size = ['None', 2]
    pool_stride = ['None', 1]

    output = ComplexConv1D(filters=filter_num[1],
                      kernel_size=kernel_size[1],
                      strides=conv_stride[1],
                      padding='same',
                      activation=None,
                      name='conv1_{}'.format(block_idx), **convArgs)(input)
    output = Modrelu(name="ModRelu1_{}".format(block_idx))(output)

    output = ComplexConv1D(filters=filter_num[2],
                      kernel_size=kernel_size[2],
                      strides=conv_stride[2],
                      padding='same',
                      activation=None,
                      name='conv2_{}'.format(block_idx), **convArgs)(output)
    output = Modrelu(name="ModRelu2_{}".format(block_idx))(output)

    # o = utils.GetAbs(name='Abs_{}'.format(block_idx))(o)

    # output = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride[1], padding='same',
    #                     name='pool_{}'.format(block_idx))(o)

    return output


def createBaseline_modrelu(inp_shape, classes_num=5, emb_size = 64, weight_decay = 1e-4, classification=False):
    dense_layer_size = ['None', 256, 256, 128]
    act_func = ['None', 'relu', 'relu', 'relu']

    blockNum = 4
    input_data = Input(shape=inp_shape)
    for i in range(blockNum):
        idx = i + 1
        if 0 == i:
            model = complex_baselineBlock(input_data, weight_decay, idx)
        else:
            model = complex_baselineBlock(model, weight_decay, idx)

    model = utils.GetAbs(name="Abs")(model)
    middle = GlobalAveragePooling1D()(model)
    dense_layer = Dense(dense_layer_size[1], name='dense1', activation=act_func[1])(middle)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation=act_func[2])(dense_layer)
    dense_layer = Dense(dense_layer_size[3], name='dense3', activation=act_func[3])(dense_layer)


    if classification:
        dense_layer = Dense(classes_num, name='dense4_classification', activation='softmax')(dense_layer)
    else:
        dense_layer = Dense(emb_size, name='dense4_feature', activation='softmax')(dense_layer)
    x = Model(inputs=input_data, outputs=dense_layer)
    return x


def createHomegrown_modrelu(inp_shape, classes_num=5, emb_size = 64, weight_decay = 1e-4, classification=False):
    convArgs = dict(use_bias=False,
                    kernel_regularizer=l2(weight_decay),
                    spectral_parametrization=False,
                    kernel_initializer='complex')

    kernel_size = ['None', 7, 5]
    filter_num = ['None', 128, 128]
    conv_stride_size = ['None', 1, 1]
    dense_layer_size = ['None', 256, 80]

    data_input = Input(shape=inp_shape)
    o = ComplexConv1D(filters=filter_num[1],
                      kernel_size=kernel_size[1],
                      strides=conv_stride_size[1],
                      padding='same',
                      activation=None,
                      name="ComplexConv1", **convArgs)(data_input)
    o = Modrelu(name="ModRelu1")(o)
    o = Dropout(0.5, name='block1_dropout')(o)

    o = ComplexConv1D(filters=filter_num[2],
                      kernel_size=kernel_size[2],
                      strides=conv_stride_size[2],
                      padding='same',
                      activation=None,
                      name="ComplexConv2", **convArgs)(o)
    o = Modrelu(name="ModRelu2")(o)
    o = Dropout(0.5, name='block2_dropout')(o)

    o = utils.GetAbs(name="Abs")(o)

    o = GlobalAveragePooling1D(name="Avg")(o)

    if classification:
        x = Dense(dense_layer_size[1],
                  activation='relu',
                  name="Dense1")(o)
        x = Dense(dense_layer_size[2],
                  activation='relu',
                  name="Dense2")(x)
        x = Dense(classes_num,
                  activation='softmax',
                  name="Dense3_classification")(x)
    else:
        x = Dense(dense_layer_size[1],
                  activation='relu',
                  name="Dense1")(o)
        x = Dense(dense_layer_size[2],
                  activation='relu',
                  name="Dense2")(x)
        x = Dense(emb_size,
                  activation='softmax',
                  name="Dense3_feature")(x)
    x = Model(inputs=data_input, outputs=x)
    return x


def createSB_modrelu(inp_shape, classes_num=5, emb_size = 64, weight_decay = 1e-4, classification=False):

    convArgs = dict(use_bias=False,
                    kernel_regularizer=l2(weight_decay),
                    spectral_parametrization=False,
                    kernel_initializer='complex')
    filters = 100
    k_size = 20
    strides = 10

    data_input = Input(shape=inp_shape)

    o = ComplexConv1D(filters=filters,
                      kernel_size=[k_size],
                      strides=strides,
                      padding='valid',
                      activation=None,
                      name="ComplexConv1", **convArgs)(data_input)

    o = Modrelu(name="ModRelu1")(o)

    filters = 100
    k_size = 10
    strides = 1
    o = ComplexConv1D(filters=filters,
                      kernel_size=[k_size],
                      strides=strides,
                      padding='valid',
                      activation=None,
                      name="ComplexConv2", **convArgs)(o)
    o = Modrelu(name="ModRelu2")(o)

    o = utils.GetAbs(name="Abs")(o)

    o = GlobalAveragePooling1D(name="Avg")(o)

    neuron_num = 100
    o = Dense(neuron_num,
              activation='relu',
              kernel_initializer="he_normal",
              kernel_regularizer=l2(weight_decay),
              name="Dense1")(o)

    if classification:
        x = Dense(classes_num,
                  activation='softmax',
                  kernel_initializer="he_normal",
                  name="Dense2_classification")(o)
    else:
        x = Dense(emb_size,
                  kernel_initializer="he_normal",
                  name="Dense2_feature")(o)

    x = Model(inputs=data_input, outputs=x)
    return x


def create_ConvLstm(inp_shape, class_num = 5, emb_size = 64, classification = False):

    kernel_size = ['None', 7, 7]
    filter_num = ['None', 50, 50]
    conv_stride_size = ['None', 1, 1]


    input_data = Input(shape=inp_shape)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], strides=conv_stride_size[1], padding='same', name='conv1')(input_data)
    model = MaxPooling1D()(model)
    model = Dropout(rate=0.5)(model)

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], strides=conv_stride_size[1], padding='same', name='conv2')(model)
    model = MaxPooling1D()(model)
    model = Dropout(rate=0.5)(model)

    model = LSTM(units=128, activation='relu', return_sequences=True, recurrent_activation='relu')(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu', name='dense1', use_bias=False)(model)

    if classification:
        dense_layer = Dense(class_num, name='dense2_classification', activation='softmax')(model)
    else:
        dense_layer = Dense(emb_size, name='dense2_feature', kernel_initializer="he_normal")(model)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def create2DHomegrown(inp_shape, class_num = 5, emb_size = 64, classification = False):
    input_data = Input(shape=inp_shape)

    kernel_size = ['None', (3,3), (3,3)]
    filter_num = ['None', 8, 8]
    conv_stride_size = ['None', 1, 1]
    pool_stride_size = ['None', 1, 1]
    activation_func = ['None', 'relu', 'relu']
    # activation_func = ['None', 'elu', 'elu']
    dense_layer_size = ['None', 256, 80]

    model = Conv2D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = Activation(activation_func[1], name='block1_act1')(model)
    model = Dropout(0.5, name='block1_dropout')(model)

    model = Conv2D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation(activation_func[2], name='block2_act1')(model)
    model = Dropout(0.5, name='block2_dropout')(model)

    output = GlobalAveragePooling2D()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation='relu')(output)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation='relu')(dense_layer)
    if classification:
        dense_layer = Dense(class_num, name='dense3_classification', activation='softmax')(dense_layer)
    else:
        dense_layer = Dense(emb_size, name='dense3_feature', kernel_initializer="he_normal")(dense_layer)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2

def createHomegrown(inp_shape, class_num = 5, emb_size = 64, classification = False):
    # -----------------Entry flow -----------------

    input_data = Input(shape=inp_shape)

    kernel_size = ['None', 7, 7]
    filter_num = ['None', 50, 50]
    conv_stride_size = ['None', 1, 1]
    pool_stride_size = ['None', 1, 1]
    activation_func = ['None', 'relu', 'relu']
    #activation_func = ['None', 'elu', 'elu']
    dense_layer_size = ['None', 256, 80]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                 strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = Activation(activation_func[1], name='block1_act1')(model)
    model = Dropout(0.5, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                 strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation(activation_func[2], name='block2_act1')(model)
    model = Dropout(0.5, name='block2_dropout')(model)

    output = GlobalAveragePooling1D()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation='relu')(output)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation='relu')(dense_layer)
    if classification:
        dense_layer = Dense(class_num, name='dense3_classification', activation='softmax')(dense_layer)
    else:
        dense_layer = Dense(emb_size, name='dense3_feature', kernel_initializer="he_normal")(dense_layer)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2


def baselineBlock(input, block_idx):

    kernel_size = ['None', 7, 5]
    filter_num = ['None', 128, 128]
    conv_stride = ['None', 1, 1]
    pool_size = ['None', 2]
    pool_stride = ['None', 1]
    act_func = 'relu'

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], name='conv1_{}'.format(block_idx),
                 strides=conv_stride[1], padding='same', activation=act_func)(input)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2], name='conv2_{}'.format(block_idx),
                 strides=conv_stride[2], padding='same', activation=act_func)(model)
    output = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride[1], padding='same',
                        name='pool_{}'.format(block_idx))(model)

    return output


def createBaseline(inp_shape, class_num = 5, emb_size = 64, classification = False):

    dense_layer_size = ['None', 256, 256, 128]
    act_func = ['None', 'relu', 'relu', 'relu']

    blockNum = 4
    input_data = Input(shape=inp_shape)
    for i in range(blockNum):
        idx = i + 1
        if 0 == i:
            model = baselineBlock(input_data, idx)
        else:
            model = baselineBlock(model, idx)

    middle = GlobalAveragePooling1D()(model)

    dense_layer = Dense(dense_layer_size[1], name='dense1', activation=act_func[1])(middle)
    dense_layer = Dense(dense_layer_size[2], name='dense2', activation=act_func[2])(dense_layer)
    dense_layer = Dense(dense_layer_size[3], name='dense3', activation=act_func[3])(dense_layer)

    if classification:
        dense_layer = Dense(class_num, name='dense4_classification', activation='softmax')(dense_layer)
    else:
        dense_layer = Dense(emb_size, name='dense4_feature', kernel_initializer="he_normal")(dense_layer)
    conv_model = Model(inputs=input_data, outputs=dense_layer)
    return conv_model


def createResnet(inp_shape, class_num = 5, emb_size = 64, classification = False):
    import resnet50_1D as resnet50
    return resnet50.create_model(inp_shape, emb_size)


def createAFtest(inp_shape, class_num = 5, emb_size = 64):
    inp, embedding = af_model.build_embedding_Homegrown(inp_shape, emb_size)
    param = af_classifier.generate_params()
    model = af_model.Expand_Dim_Layer(embedding, 'cls_conv')
    model = Conv1D(filters=param['cls_conv_{}'.format(1)], kernel_size=param['cls_kernel_{}'.format(1)],
                   strides=1, padding='same', name='class_conv_{}_{}'.format(1, 1))(model)
    model = BatchNormalization(name='class_bn_{}_{}'.format(1, 1))(model)
    model = Activation(param['cls_act'], name='class_act_{}_{}'.format(1, 1))(model)
    model = MaxPooling1D(pool_size=param['cls_pool_{}'.format(1)], strides=1,
                         padding='same', name='class_pool_{}_{}'.format(1, 1))(model)
    model = Dropout(param["drop_classifier"], name='class_drop_{}_{}'.format(1, 1))(model)

    model = GlobalAveragePooling1D()(model)

    dense2 = Dense(param['cls_dense2'], name='class_dense2')(model)
    bn2 = BatchNormalization(name='class_bn2')(dense2)
    act2 = Activation(param['cls_act'], name='class_act2')(bn2)
    drop2 = Dropout(param["drop_classifier"], name='class_drop2')(act2)

    densel = Dense(class_num, name='class_dense_last')(drop2)
    bnl = BatchNormalization(name='class_bn_last')(densel)
    actl = Activation('softmax', name='class_act_last')(bnl)
    m = Model(inputs=inp, outputs = actl)

    return m


def create_model(modelType, inp_shape, NUM_CLASS, emb_size, classification):

    print("model type: {}".format(modelType))

    if 'homegrown' == modelType:
        model = createHomegrown(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif '2Dhomegrown' == modelType:
        model = create2DHomegrown(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'baseline' == modelType:
        model = createBaseline(inp_shape, NUM_CLASS, emb_size,classification=classification)
    elif 'resnet' == modelType:
        model = createResnet(inp_shape, NUM_CLASS, emb_size,classification=classification)
    elif 'complex' == modelType:
        model = createSB_modrelu(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'complex_homegrown' == modelType:
        model = createHomegrown_modrelu(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'complex_baseline' == modelType:
        model = createBaseline_modrelu(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'complex_DF' == modelType:
        model = create_ComplexDF(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'complex_convnet' == modelType:
        model = create_convnet()
    elif 'AF' == modelType:
        model = createAFtest(inp_shape, NUM_CLASS, emb_size)
    elif 'lstm' == modelType:
        model = create_ConvLstm(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif 'DF' == modelType:
        model = create_DF(inp_shape, NUM_CLASS, emb_size, classification=classification)
    elif '2DDF' == modelType:
        model = create_2DDF(inp_shape, NUM_CLASS, emb_size, classification=classification)
    else:
        raise ValueError('model type {} not support yet'.format(modelType))

    return model


def test_run(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')


def test():
    modelTypes = ['complex_convnet']
    NUM_CLASS = 10
    signal = True
    inp_shape = (2, 288)
    emb_size = 64
    for modelType in modelTypes:
        model = create_model(modelType, inp_shape, NUM_CLASS, emb_size, classification=True)
        try:
            flag = test_run(model)
        except Exception as e:
            print(e)

    print('all done!') if signal else print('test failed')


if __name__ == "__main__":
    test()
