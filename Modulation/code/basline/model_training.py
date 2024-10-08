#!/usr/bin/python3
from pathlib import Path
import sys
from datetime import datetime 
import numpy as np
sys.path.append('models')
import rf_models
#import wandb
#from wandb.integration.keras import WandbCallback
from typing import Dict, List, Tuple
import h5py
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import shuffle
import h5py as h5
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation, GlobalAveragePooling1D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.layers import Reshape, Dense, Flatten, Add
from tensorflow.keras.activations import relu


path = Path('/home/mabon/TinyRadio/Modulation/data/2021RadioML.hdf5')
name = 'radioml'
#new_exp='/home/mabon/TinyRadio/Modulation/experiments/ResNet/2021_27/BASELINE/'
new_exp='/home/mabon/RYAN_TRAING_SHIT_MOD_BASELINE'
n2_best = '/home/mabon/TinyRadio/Modulation/experiments/ResNet/2021_27/pruned/l2/automatic/N2/best_checkpoint.h5'
threshold = 6

os.makedirs(new_exp, exist_ok=True)
def plot_confusion_matrix(cm,path, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize = (15,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    label_len = np.shape(labels)[0]
    tick_marks = np.arange(label_len)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)  # You can change the filename and extension as needed
def resnet_block(input_data, in_filters, out_filters, conv_size):
  x = Conv2D(in_filters, conv_size, activation=None, padding='same')(input_data)
  x = BatchNormalization()(x)
#   x = Add()([x, input_data])
  x = Activation('relu')(x)
  x = Conv2D(out_filters, conv_size, activation=None, padding='same')(x) 
  x = BatchNormalization()(x)
  x = MaxPooling2D(2, strides = (2,1), padding = 'same') (x)
  return x


def main():
    # Initialize wandb
    

    # Setup params
    Batch_Size = 1024
    # lr = 0.1


    os.makedirs(os.path.join(new_exp), exist_ok=True)
    file_handle = h5.File(path,'r+')

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

    print(f"HERE:: {np.shape(Y_test)}")
    return

    classes = mods
    del myData, myMods, mySNRs
    num_resnet_blocks = 5
    kernel_size = 5,1
    input_shp = list(X_train.shape[1:])
    rf_input = Input(shape=input_shp, name = 'rf_input')

    x = Conv2D(16, (kernel_size), activation=None, padding='same')(rf_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    in_filters = 16
    out_filters = 32
    for i in range(num_resnet_blocks):
        if (i == num_resnet_blocks-1):
            out_filters = num_classes
        x = resnet_block(x, in_filters, out_filters, kernel_size)
        in_filters = in_filters * 2
        out_filters = out_filters * 2

    flatten = Flatten()(x)
    dropout_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(num_classes, activation='relu')(dropout_1)        
    softmax = Activation('softmax', name = 'softmax')(dense_1)

    optimizer= Adam(learning_rate=0.00060)
    model = keras.Model(rf_input, softmax)
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"])

    print(model.summary())


    print("LOG----------------------DATALOAD----------------------")


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =new_exp + '/best_checkpoint.h5', 
                                                    verbose = 1,
                                                    save_best_only=True, 
                                                    save_weights_only=False,
                                                    mode='auto')
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            mode='auto',
            verbose = 1)

    start = datetime.now()

    # Train the model
    #with tf.device('/GPU:0'):            
    #    history = model.fit(
    #        X_train,
    #        Y_train,
    #        batch_size=Batch_Size,
    #        epochs=150,
    #        verbose=1,
    #        validation_data=(X_val, Y_val),
    #        callbacks = [ cp_callback,earlystopping_callback]
    #        )
    runtime = datetime.now() - start 

    print(f"[RUNTIME] The training run time was: {runtime.total_seconds()} seconds")
    #print(f"[RUNTIME] The training epoch: {len(history.history['loss'])} epochs")
    #print(f"[RUNTIME] The sec per epoch: {runtime.total_seconds()/len(history.history['loss'])} sec per epochs")

    #best_checkpoint = new_exp + '/best_checkpoint.h5'
    # model.load_weights(best_checkpoint)


    best_checkpoint = n2_best
    model = tf.keras.models.load_model(best_checkpoint)

    # Show simple version of performance
    score = model.evaluate(X_test, Y_test,  verbose=0, batch_size=Batch_Size)
    print(score)
    # Plot confusion matrix
    start = datetime.now()
    test_Y_hat = model.predict(X_test, batch_size=Batch_Size)
    runtime = datetime.now() - start
    print(f"The runtime was {runtime.total_seconds()}")
    print(f"The frames were {X_test.shape} shape and {X_test.shape[0]}")
    print(f"The frame per sec was {X_test.shape[0]/runtime.total_seconds()} fps")

    conf = np.zeros([num_classes,num_classes])
    confnorm = np.zeros([num_classes,num_classes])
    for i in range(0,X_test.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,num_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    saveplotpath=new_exp+'/matrix.png'    
    plot_confusion_matrix(confnorm,saveplotpath, labels=classes)
    # Predict and calculate classification report
    Y_pred = model.predict(X_test, batch_size=Batch_Size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(Y_test, axis=1)
    classification_report_fp = classification_report(y_actual, y_pred, target_names=mods)

    # Print the classification report
    print(classification_report_fp)
    report_path = os.path.join(new_exp, 'classification_report.txt')

    # Save the classification report to a file
    with open(report_path, 'w') as file:
        file.write(classification_report_fp)

    print(f"Classification report saved to {report_path}")
    # Convert same_day_score to string
    same_day_score_str = str(score)

    # Write the string to the file
    with open(os.path.join(new_exp, 'Accuracylos.txt'), "w") as file:
        file.write(same_day_score_str)

if __name__ == "__main__":
    main()
