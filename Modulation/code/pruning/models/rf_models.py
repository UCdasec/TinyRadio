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
#from complexnn import ComplexConv1D, Modrelu, SpectralPooling1D, ComplexDense
#from complexnn import utils
import sys
sys.path.append('ADA')



def createResnet(inp_shape, class_num = 5, emb_size = 64,pruning_rate=1, classification = False):
    import resnet50_1D as resnet50
    return resnet50.create_model(inp_shape, emb_size,pruning_rate)



def create_model(modelType, inp_shape, NUM_CLASS, emb_size, pruning_rate,classification):

    print("model type: {}".format(modelType))


    if 'resnet' == modelType:
        model = createResnet(inp_shape, NUM_CLASS, emb_size,pruning_rate,classification=classification)
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
