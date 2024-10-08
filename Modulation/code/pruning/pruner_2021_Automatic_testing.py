#!/usr/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import h5py as h5
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import argparse
sys.path.append('models')



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
    ranks = list()
    testY = np.array(testY)
    for i in range(num_class):
        ids = np.where(testY == i)
        if preds is None:
            OneClsData = np.squeeze(np.take(testX, ids, axis=0))
            OneClsData = np.reshape(OneClsData, (-1, 1024, 1, 2))  # Adjust the reshape as needed
            OneClsPreds = model.predict(OneClsData, verbose=1)
        else:
            OneClsPreds = np.take(preds, ids, axis=0)
        OneCls_rank = accumulate_ranks(np.squeeze(OneClsPreds), num_class, i)

        ranks.append(OneCls_rank)
    return ranks




def main(opts):
    # Initialize wandb
    #wandb.init(project="RadioFingerPrint", entity="mabonmn2002")
    os.makedirs(opts.new_exp, exist_ok=True)

    # Setup params
    os.makedirs(os.path.join(opts.new_exp), exist_ok=True)
    file_handle = h5.File(opts.path,'r+')

    new_myData = file_handle['X'][:]  #1024x2 samples 
    new_myMods = file_handle['Y'][:]  #mods 
    new_mySNRs = file_handle['Z'][:]  #snrs  

    file_handle.close()
    myData = []
    myMods = []
    mySNRs = []
    # Define the threshold
    threshold = 6
    for i in range(len(new_mySNRs)):
        if new_mySNRs[i]>=threshold:
            myData.append(new_myData[i])
            myMods.append(new_myMods[i])
            mySNRs.append(new_mySNRs[i])
    # Convert lists to NumPy arrays
    myData = np.array(myData)
    myMods = np.array(myMods)
    mySNRs = np.array(mySNRs)          
    # Print the shapes of the new arrays
    print(np.shape(myData))
    print(np.shape(myMods))
    print(np.shape(mySNRs))
    myData = myData.reshape(myData.shape[0], 1024, 1, 2) 

    mods = ["OOK","4ASK","8ASK",
            "BPSK","QPSK","8PSK","16PSK","32PSK",
            "16APSK","32APSK","64APSK","128APSK",
            "16QAM","32QAM","64QAM","128QAM","256QAM",
            "AM-SSB-WC","AM-SSB-SC","AM-DSB-WC","AM-DSB-SC","FM",
            "GMSK","OQPSK","BFSK","4FSK","8FSK"]

    num_classes=27

    # First split: 80% train, 20% temp (test + validation)
    X_train, X_temp, Y_train, Y_temp, Z_train, Z_temp = train_test_split(
        myData, myMods, mySNRs, test_size=0.2, random_state=0)

    # Second split: 50% of the temp data for validation, 50% for testing (since it's 10% of the original data)
    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(
        X_temp, Y_temp, Z_temp, test_size=0.5, random_state=0) 
    print (np.shape(X_test))
    print (np.shape(Y_test))
    print (np.shape(Z_test))
    print (np.shape(X_train))
    print (np.shape(Y_train))
    print (np.shape(Z_train))
    del myData, myMods, mySNRs
    inp_shape = list(X_train.shape[1:])
    print("Dataset Shape={0} CNN Model Input layer={1}".format(X_train.shape, inp_shape))
    classes = mods
    print("The number of classes is ", classes)
  
    modelPath = opts.new_exp + '/best_checkpoint.h5'
    m = load_model(modelPath)
    m.summary()
    NUM_CLASS=27  

    _, acc = m.evaluate(X_test, Y_test, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))

    ranks = class_ranks(m, X_test, Y_test, NUM_CLASS, preds=None)
    total_rank = np.array(ranks)
    rawtrue = os.path.join(opts.new_exp, 'DeviceRank.csv')

    df = pd.DataFrame(total_rank, index=None)
    df.to_csv(rawtrue,header=False)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', help='')
    parser.add_argument('-o', '--new_exp', help='')
    opts = parser.parse_args()
    return opts

    
if __name__ == "__main__":
    opts = parseArgs(sys.argv)

    main(opts)

