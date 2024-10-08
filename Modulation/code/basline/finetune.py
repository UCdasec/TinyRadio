#! /usr/bin/env python3.6

import os
import sys
import argparse
import time
import random
from statistics import mean, stdev
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import numpy as np
from collections import defaultdict
import rf_models
import load_slice_IQ
import augData

import tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)

# thresholdList = 0.3 - 1 / np.logspace(0.55, 2, num=25, endpoint=True)
thresholdList = 0.4 - 1 / np.logspace(0.4, 2, num=25, endpoint=True)


class CNN():
    def __init__(self, opts, dataOpts):
        self.verbose = opts.verbose
        self.trainData = opts.trainData
        self.tuneData = opts.tuneData
        self.trainModelPath = os.path.join(modelDir, 'train_best_{}_{}_sampled:{}'.format(opts.modelType, os.path.basename(opts.trainData), dataOpts.sample))
        self.batch_size = 128
        self.trainEpochs = 100
        self.tuneEpochs = 30
        self.dataOpts = dataOpts
        self.report = []
        self.input_shape = 5
        self.count = 0



    def createModel(self, NUM_CLASS, topK=False):
        print("load well trained model")
        # model = DF(input_shape=input_shape, emb_size=emb_size, Classification=True)
        model = rf_models.create_model(opts.modelType, inp_shape=self.input_shape, NUM_CLASS=NUM_CLASS, emb_size=64, classification=True)
        print('model compiling...')
        metricList = ['accuracy']
        if topK:
            metricList.append('top_k_categorical_accuracy')
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=metricList)
        return model

    def train(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        '''train the cnn model'''
        model = self.createModel(NUM_CLASS)

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.trainModelPath, monitor='val_accuracy', verbose=1,
                                       save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=30)
        callBackList = [checkpointer, earlyStopper]

        start = time.time()
        model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.trainEpochs,
                  validation_split=0.1,
                  verbose=self.verbose,
                  callbacks=callBackList,
                  shuffle=True)
        end = time.time()
        time_last = end - start
        print('Testing with best model...')
        m = load_model(self.trainModelPath)
        score, acc = m.evaluate(X_test, y_test, batch_size=self.batch_size)
        reportLine = 'Test accuracy with data {} is: {:f}\n'.format(self.trainData, acc)
        print(reportLine)
        return reportLine, time_last

    def tuneTheModel(self, X_train, y_train, NUM_CLASS):
        self.tuneModelPath = os.path.join(modelDir, 'tune_trsn_{}'.format(y_train.shape[1]))
        old_model = load_model(self.trainModelPath, compile=False)
        new_model = self.createModel(NUM_CLASS, topK=True)

        print("copying weights from old model to new model...")
        LayNum = len(new_model.layers) - 3
        for l1, l2 in zip(new_model.layers[:LayNum], old_model.layers[:LayNum]):
            l1.set_weights(l2.get_weights())
            l1.trainable = False

        print('Fitting model...')
        checkpointer = ModelCheckpoint(filepath=self.tuneModelPath, monitor='val_accuracy', verbose=1,
                                       save_best_only=True, mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
        callBackList = [checkpointer, earlyStopper]

        hist = new_model.fit(X_train, y_train,
                             batch_size=self.batch_size,
                             epochs=self.tuneEpochs,
                             validation_split=0.1,
                             verbose=0,
                             callbacks=callBackList)
        return new_model

    def test_close(self, new_model, X_test, y_test):
        self.count += 1
        print(self.count)
        print('Testing with best model...')
        score, acc, top5Acc = new_model.evaluate(X_test, y_test, batch_size=100)
    
        from utility import class_ranks, ranks_KNN
        y=list()
        for i in y_test:
            y.append(np.argmax(i))
        ranks = class_ranks(new_model, X_test, y, 20, preds=None)
        #ranks = ranks_KNN(y_test, preds, 20)
        #if i == 0:
        total_rank = np.array(ranks)
        #else:
        #    total_rank += np.array(ranks)
        #    print(total_rank.shape)

        df = pd.DataFrame(total_rank, index=None)
        df.to_csv("rank_result/finetune_neu_" + str(self.count) + '.csv',header=False)
        reportLine = 'Test accuracy of tune model with data {} is: {:f}, and test top 5 acc is: {:f}\n'.format(
            self.tuneData, acc, top5Acc)
        print(reportLine)
        return acc, top5Acc

    def tune(self, X_train, y_train, X_test, y_test, NUM_CLASS):
        self.input_shape = (X_train.shape[1], X_train.shape[2])
        new_model = self.tuneTheModel(X_train, y_train, NUM_CLASS)
        acc, top5Acc = self.test_close(new_model, X_test, y_test)
        return acc, top5Acc



def prepareData(X, y, trsn=40, test=10):
    dataDict = defaultdict(list)
    for i in range(len(y)):
        oneLabel = y[i]
        oneData = X[i, :]
        dataDict[oneLabel].append(oneData)

    X_train, X_test, y_train, y_test = [], [], [], []
    NUM_CLASS = len(list(dataDict.keys()))
    for key in dataDict.keys():
        oneClsData = dataDict[key]
        random.shuffle(oneClsData)
        # split train and test
        train_samples = oneClsData[:trsn]
        test_samples = oneClsData[trsn:trsn + test]

        train_labels = np.ones(len(train_samples), dtype=np.int) * int(key)
        test_labels = np.ones(len(test_samples), dtype=np.int) * int(key)

        X_train.extend(train_samples)
        y_train.extend(train_labels)
        X_test.extend(test_samples)
        y_test.extend(test_labels)

    # shuffle data
    X_train, y_train = mytools.shuffleData(X_train, y_train)
    X_test, y_test = mytools.shuffleData(X_test, y_test)

    y_train = to_categorical(y_train, NUM_CLASS)
    y_test = to_categorical(y_test, NUM_CLASS)
    return X_train, y_train, X_test, y_test, NUM_CLASS


def main(opts, dataOpts):
    model = CNN(opts, dataOpts)
    source = os.path.basename(opts.trainData).split('.')[0]
    target = os.path.basename(opts.tuneData).split('.')[0]
    test_times = 3
    rtnLine = ''

    if opts.trainData and (not opts.modelPath):
        print('train the model once...')

        X_train, y_train, X_test, y_test, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
        y_train = to_categorical(y_train, NUM_CLASS)
        y_test = to_categorical(y_test, NUM_CLASS)
        print('train data shape: ', X_train.shape)
        rtnLine, time_last = model.train(X_train, y_train, X_test, y_test, NUM_CLASS)
        print(rtnLine)
        del X_train, y_train, X_test, y_test, NUM_CLASS

    if opts.modelPath:
        model.trainModelPath = opts.modelPath

    if 'tsn' == opts.testType:
        print('start run n_shot test...')
        trsnList = [100,200,400,800]
        tesnList = [2000] * len(trsnList)
        snList = zip(trsnList, tesnList)
        # tsnList = [5]
        resultFile = os.path.join(ResDir, 'tune_model_{}_to_{}.txt'.format(source, target))
        f = open(resultFile, 'a+')
        print('\n\n##################### test time is: {} ####################'.format(time.ctime()), flush=True,
              file=f)
        X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        yy = to_categorical(y)
        old_m = load_model(model.trainModelPath)
        _, ori_acc = old_m.evaluate(X, yy, batch_size=100, verbose=1)
        rtnLine += 'ori_acc: {:f}'.format(ori_acc)
        for sn in snList:
            # opts.nShot = trsn
            acc_list, acc_top5_list = [], []
            for i in range(test_times):
                X_train, y_train, X_test, y_test, NUM_CLASS = prepareData(X, y, sn[0], sn[1])

                print('tune data shape: ', X_train.shape)
                print('tune the model...')
                acc, acctop5 = model.tune(X_train, y_train, X_test, y_test, NUM_CLASS)
                acc_list.append(acc)
                acc_top5_list.append(acctop5)
            rtnLine = rtnLine + 'model: {}, stride={}, trsn={}, test={}, tune model with data {}, acc is: {:f}, and std is: {:f}\n'.format(
                    opts.modelType, dataOpts.stride, sn[0], sn[1], model.tuneData, mean(acc_list), stdev(acc_list))
        print(rtnLine, file=f, flush=True)
        f.close()


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--trainData', default='', help='file path of config file')
    parser.add_argument('-tu', '--tuneData', help='file path of config file')
    # parser.add_argument('-o', '--openData', help ='file path of open data file')
    parser.add_argument('-ns', '--nShot', type=int, help='n shot number')
    parser.add_argument('-m', '--modelPath', default='', help='file path of open data file')
    parser.add_argument('-d', '--data_dim', type=int, default=1500, help='file path of config file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose or not')
    parser.add_argument('-a', '--augData', type=int, default=0, help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='tsn', help='choose different test: tsn/aug/trainNum/trainTime')
    parser.add_argument('-pf', '--prefix', help='')
    ######################## rf args ###############
    parser.add_argument('-i', '--input', help='input file/dir')
    parser.add_argument('-o', '--output', help='output dir')
    parser.add_argument('-mt', '--modelType', default='homegrown', help='choose from homegrown/baseline/resnet')
    parser.add_argument('-sp', '--splitType', default='random', help='choose from random/order')
    parser.add_argument('--D2', action='store_true', help='if set will return 2 dimension data')
    parser.add_argument('-n', '--normalize', action='store_true', help='')
    parser.add_argument('-ds', '--dataSource', help='choose from neu/simu')
    parser.add_argument('-cf', '--channel_first', action='store_true',
                        help='if set channel first otherwise channel last')
    parser.add_argument('-l', '--location', help='data collected location')

    opts = parser.parse_args()
    return opts


class testOpts():
    def __init__(self, trainData, tuneData, location, testType, modelType, modelPath=None, start_idx=0):
        self.trainData = trainData
        self.tuneData = tuneData
        self.testType = testType
        self.modelPath = modelPath
        self.location = location
        self.verbose = True
        self.modelType = modelType
        self.augData = False
        self.start_idx = start_idx


if __name__ == "__main__":
    # opts = parseOpts(sys.argv)

    opts = testOpts(trainData='/home/haipeng/Documents/dataset/radio_dataset/neu_different_day3',
                    tuneData='/home/haipeng/Documents/dataset/radio_dataset/neu_different_day4',
                    location='after_equ',
                    testType='tsn',
                    modelType='DF',
                    modelPath = '/home/haipeng/Documents/radio_fingerprinting/rank_models/IQ_neu_different_day3_DF_slices_10000_startIdx_200000_sample_False_mul_trans_True_0_0.15'
                    )
    # if opts.useGpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dataOpts = load_slice_IQ.loadDataOpts(opts.tuneData, opts.location, num_slice=10000, slice_len=288,
                                       start_idx = 200000, stride = 288, mul_trans = True, window = 64, dataType='IQ')

    main(opts, dataOpts)
