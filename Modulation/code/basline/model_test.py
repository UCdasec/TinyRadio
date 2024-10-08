import os

import numpy as np
import pandas as pd
from statistics import mean, stdev
import load_slice_IQ
sys.path.append('models')

import tools as mytools
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from utility import class_ranks
import time
new_exp='/home/mabon/RadioFingerPrinting/TEST/NEW_TEST/afterfft'
ResDir = os.path.join(new_exp, 'res_out')
os.makedirs(ResDir, exist_ok=True)



def single_model_test():

    modelPath = '/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/best_model_neu_different_day3_DF_10000_sample_False_mul_trans_True_0_0.32'
    m = load_model(modelPath)
    m.summary()
    dataPath = "/home/haipeng/Documents/dataset/radio_dataset/neu_different_day4"
    opts = tfOpts(source_path=dataPath, location='after_equ')
    dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location,  num_slice = 10000, slice_len=opts.slice_len, start_idx = 0, sample = False, mul_trans = True)
    _, _, X, y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
    test_y = to_categorical(y, NUM_CLASS)
    test_x, test_y = mytools. shuffleData(X, test_y)

    score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))


def rank_model_test():

    modelPath = '/home/haipeng/Documents/radio_fingerprinting/res_out/modelDir/best_model_neu_different_day3_DF_10000_sample_True_mul_trans_True_0_0.05'
    m = load_model(modelPath)
    m.summary()
    dataPath = "/home/haipeng/Documents/dataset/radio_dataset/neu_different_day3"
    opts = tfOpts(source_path=dataPath, location='after_equ')
    dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location, num_slice = 1000, slice_len=288, stride=576,  start_idx = 200000, sample = False, dataType='IQ')
    #test_time = 3
      
    X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
    test_y = to_categorical(y, NUM_CLASS)
    test_x, test_y = mytools. shuffleData(X, test_y)
    #for i in range(test_time):
          #_, _, X, y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
        #X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        #test_y = to_categorical(y, NUM_CLASS)
        #test_x, test_y = mytools. shuffleData(X, test_y)

    score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))

    ranks = class_ranks(m, X, y, NUM_CLASS, preds=None)
    #if i == 0:
    total_rank = np.array(ranks)
    #else:
    #    total_rank += np.array(ranks)
    #    print(total_rank.shape)

    df = pd.DataFrame(total_rank, index=None)
    name = os.path.basename(modelPath)
    df.to_csv("rank_result/sameday_rank_" + name + '.csv',header=False)

if __name__ == '__main__':
    rank_model_test()
    #ada_classifier_test()
    #model_test()
    #single_model_test()


