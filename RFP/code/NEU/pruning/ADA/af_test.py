#! /usr/bin/env python3

import os
import sys
import argparse
from statistics import mean, stdev
import time
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import af_model
import utility as utility
import tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'ADA/res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def tuning_model(inp, extractor, pre_model, feature_model, data_dict, sites):
    allData, allLabel = mytools.datadict2data(data_dict, sites)
    clsNum = len(sites)
    # allData = allData[:, :, np.newaxis]
    allLabel = to_categorical(allLabel, clsNum)

    # replace the last layer
    outLayer = Dense(clsNum, activation='softmax')(extractor)
    new_model = Model(inputs=inp, outputs=outLayer)
    new_model = copy_weights(new_model, pre_model, compileModel=False)

    print('Compiling...')
    new_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # tunning the model
    modelPath = os.path.join(ResDir, 'best_tune_model')
    checkpointer = ModelCheckpoint(filepath=modelPath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='accuracy', mode='max', patience=10)
    callBackList = [checkpointer, earlyStopper]

    new_model.fit(allData, allLabel, batch_size=64, epochs=30, verbose=1, shuffle=True, callbacks=callBackList)
    feature_model = copy_weights(feature_model, new_model)

    return feature_model, new_model


def copy_weights(feature_model, classifier, compileModel=True):
    depth = len(feature_model.layers) - 3
    for l1, l2 in zip(feature_model.layers[:depth], classifier.layers[:depth]):
        l1.set_weights(l2.get_weights())

    '''
    depth = len(feature_model.layers) - 6
    for layer in feature_model.layers:
        if depth > 0:
            layer.trainable = False
        else:
            l1.trainable = True
        depth = depth - 1
    '''

    if compileModel:
        feature_model.compile(loss='mse', optimizer=Adam())
    return feature_model


def run(opts, modelType, signature_dict, test_dict, params, emb_size, inp_shape):
    sites = list(test_dict.keys())
    type_exp = 'N-MEV' if opts.exp_type else ''

    # load model
    classifier = load_model(opts.model_path, compile=False)
    if modelType == 'homegrown':
        inp, extractor = af_model.build_embedding_Homegrown(inp_shape, emb_size)
    elif modelType == 'DF':
        inp, extractor = af_model.build_embedding_DF_model(inp_shape, emb_size)
    feature_model = Model(inputs=inp, outputs=extractor)

    start = time.time()
    if opts.tuning:
        feature_model, tuned_classifer = tuning_model(inp, extractor, classifier, feature_model, signature_dict, sites)
    else:
        feature_model = copy_weights(feature_model, classifier)

    signature_vector_dict, test_vector_dict = utility.create_test_set_Wang_disjoint(signature_dict, test_dict, sites,
                                                                                    features_model=feature_model,
                                                                                    type_exp=type_exp)
    # Measure the performance (accuracy)
    acc_knn_top1, acc_knn_top5 = utility.kNN_accuracy(signature_vector_dict, test_vector_dict, params)
    end = time.time()
    duration = end-start
    return acc_knn_top1, duration


def fine_tune_test(opts, signature_dict, test_dict, emb_size, inp_shape):
    sites = list(test_dict.keys())

    # load model
    classifier = load_model(opts.model_path, compile=False)
    inp, extractor = af_model.build_embedding_Homegrown(inp_shape, emb_size)
    feature_model = Model(inputs=inp, outputs=extractor)

    feature_model, tuned_classifer = tuning_model(inp, extractor, classifier, feature_model, signature_dict, sites)

    testX, testY = mytools.datadict2data(test_dict)
    test_y = to_categorical(testY, 5)
    test_x, test_y = mytools.shuffleData(testX, test_y)

    score, acc = tuned_classifer.evaluate(test_x, test_y, batch_size=64, verbose=1)

    print("acc: {}".format(acc))
    return acc


def generate_default_params(opts):
    params = {
        'weights': 'distance',
        'p': 2,
        'metric': 'cosine',
        'k': opts.nShot
    }
    return params


def main(opts):
    params = generate_default_params(opts)
    source = os.path.basename(opts.source).split('.')[0]
    target = os.path.basename(opts.target).split('.')[0]
    inp_shape = (288, 2)
    n_shot = opts.n_shot
    test_times = 3
    #outfile = os.path.join(ResDir, 'test_ADA_one_source_{}_target_{}_res.txt'.format(source, target))
    #f = open(outfile, 'a+')
    #print('\n\n##################### test time is: {}####################'.format(time.ctime()), file=f, flush=True)

    acc_list_Top1, acc_list_Top5 = [], []
    for i in range(test_times):
        signature_dict, test_dict, sites = utility.getRFdataDict(opts.target, opts, n_shot = n_shot, n_instance=2000+n_shot, max_n=n_shot)
        #acc = fine_tune_test(opts, signature_dict, test_dict, inp_shape=inp_shape, emb_size=64)
        #acc_list_Top1.append(acc) 
        size_of_problem = len(sites)
        print("Size of Problem: ", size_of_problem, "\tN_shot: ", n_shot)
        acc_knn_top1, acc_knn_top5 = run(opts, 'DF', signature_dict, test_dict, params, inp_shape=inp_shape, emb_size=64)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))


    #mean_top1, mean_top5 = mean(acc_list_Top1), mean(acc_list_Top5)
    #test_res = 'acc for top 1: {:f}\tacc for top 5: {:f}'.format(mean_top1, mean_top5)
    test_res = 'acc for top 1: {:f}, std: {:f}'.format(mean(acc_list_Top1), stdev(acc_list_Top1))
    #test_res = test_res + '\ntest run for {} times'.format(test_times)
    print(test_res)
    #print(test_res, file=f, flush=True)

    #f.close()

class MyOpts():
    def __init__(self, model_path, nShot=5, tuning=True, aug=0, exp_type=False):
        self.model_path = model_path
        self.nShot = nShot
        self.tuning = tuning
        self.aug = aug
        self.exp_type = exp_type


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--model_path', help='')
    parser.add_argument('-ns', '--nShot', type=int, default=5, help='')
    parser.add_argument('-tn', '--tuning', action='store_true', help='')
    parser.add_argument('-a', '--aug', type=int, default=0, help='')
    parser.add_argument('-exp', '--exp_type', action='store_true', help='')
    parser.add_argument('-dd', '--data_dim', default=5000, type=int, help='')
    opts = parser.parse_args()
    return opts


class afOpts():
    def __init__(self, source_path, test_path, location, model_path, file_key='*.bin', num_slice=100000, start_ix=0, slice_len=288, stride=1):
        self.source = source_path
        self.target = test_path
        self.model_path = model_path
        self.useGpu = True
        self.testType = "nShot"
        # self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.exp_type = False
        self.sample = False
        self.mul_trans = False
        self.tuning = True
        self.n_shot = 400


if __name__ == "__main__":
    # opts = parseOpts(sys.argv)
    opts = afOpts(source_path="/home/haipeng/Documents/dataset/radio_dataset/neu_day1",
                  test_path="/home/haipeng/Documents/dataset/radio_dataset/neu_day2",
                  location="after_equ",
                  model_path='/home/haipeng/Documents/radio_fingerprinting/ADA/res_out/modelDir/ADA_DF_neu_day1_sourceSize_10000_targetSize_400')
    main(opts)
