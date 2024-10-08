import os

import numpy as np
import pandas as pd
from statistics import mean, stdev
import load_slice_IQ

import sys
sys.path.append('models')

import tools as mytools
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import time
new_exp='/home/mabon/RadioFingerPrinting/TEST/NEW_TEST/afterfft'
ResDir = os.path.join(new_exp, 'res_out')


def accumulate_ranks(preds, num_class, gt):
    trace_num = preds.shape[0]
    rank = list()
    scores = np.zeros(num_class)
    for i in range(trace_num):
        scores += preds[i]
        r = np.argsort(scores)[::-1]
        rank.append(np.where(r==gt)[0][0])
    return rank


def class_ranks(model, testX, testY, num_class, preds=None):
    # Check the shape before reshaping
    print(f"Original shape of testX: {testX.shape}")
    
    # Ensure the data is in the expected shape
    if len(testX.shape) == 3:
        testX = testX.reshape(testX.shape[0], testX.shape[1], 1, testX.shape[2])
    elif len(testX.shape) == 4 and testX.shape[2] == 1:
        # Already in the correct shape
        pass
    else:
        raise ValueError(f"Unexpected shape of testX: {testX.shape}")

    print(f"Reshaped testX to: {testX.shape}")
    
    ranks = []
    testY = np.array(testY)
    for i in range(num_class):
        ids = np.where(testY == i)
        if preds is None:
            OneClsData = np.squeeze(np.take(testX, ids, axis=0))
            OneClsPreds = model.predict(OneClsData, verbose=1)
        else:
            OneClsPreds = np.take(preds, ids, axis=0)
        OneCls_rank = accumulate_ranks(np.squeeze(OneClsPreds), num_class, i)
        ranks.append(OneCls_rank)
    return ranks


def ranks_KNN(testY, preds, num_class):
    ranks = list()
    rank = list()
    testY = np.array(testY)
    scores = np.zeros(num_class)
    assert len(testY)==len(preds)
    for i in range(num_class):
        ids = np.where(testY == i)
        OneClsPreds = np.take(preds, ids, axis=0)[0]

        for p in OneClsPreds:
            scores[int(p)]+=1
            r = np.argsort(scores)[::-1]
            rank.append(np.where(r==i)[0][0])
        ranks.append(rank)
    return ranks


import sys
def test_close(new_model, X_test, y_test):
    print('Testing with best model...')
    
    # Ensure that X_test has the correct shape
    if len(X_test.shape) == 3:  # (batch, 864, 2)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, X_test.shape[2])
    elif len(X_test.shape) == 4 and X_test.shape[2] == 1:
        pass  # Already in the correct shape
    else:
        raise ValueError(f"Unexpected shape of X_test: {X_test.shape}")

    print(f"Reshaped X_test to: {X_test.shape}")

    # Evaluate the model
    score, acc = new_model.evaluate(X_test, y_test, batch_size=100)
    print(f"Test score: {score}, Test accuracy: {acc}")
    
    # Convert y_test to class labels
    y = [np.argmax(i) for i in y_test]
    ranks = class_ranks(new_model, X_test, y, 20, preds=None)

    total_rank = np.array(ranks)
    df = pd.DataFrame(total_rank, index=None)
    df.to_csv("finetune_neu_" + str(count) + '.csv', header=False)
    
    reportLine = f'Test accuracy of tuned model is: {acc:.4f}'
    print(reportLine)
    return acc





def rank_model_test():

    modelPath = '/home/mabon/TinyRadio/RFP/experiments/ResNet/NEU/BASELINE/best_checkpoint.h5'
    m = load_model(modelPath)
    m.summary()
    dataPath = "/home/mabon/NEU/our_day1"
    dataOpts = load_slice_IQ.loadDataOpts(dataPath, 'after_equ', num_slice = 5000, slice_len=864, stride=864,  start_idx = 50000, sample = False, dataType='IQ')
    #test_time = 3
      
    X, y,_,_, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)

    test_x  = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
    test_y = to_categorical(y, num_classes=20)

    test_close(m, test_x, test_y)

if __name__ == '__main__':
    rank_model_test()
    #ada_classifier_test()
    #model_test()
    #single_model_test()


