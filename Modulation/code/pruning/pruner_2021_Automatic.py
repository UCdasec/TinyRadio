#!/usr/bin/python3
"""
Author: Mabon Manoj
"""

from datetime import datetime
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py as h5
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Input,
    BatchNormalization, Dropout,
    Activation,
    Dense, Flatten
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse

sys.path.append('models')

def resnet_block_auto(input_data, in_filters, out_filters, conv_size, r,Counter):
    print(r[Counter])
    x = Conv2D(int(in_filters * r[Counter]), conv_size, activation=None, padding='same')(input_data)
    x = BatchNormalization()(x)
    Counter+=1
    print(r[Counter])
    x = Activation('relu')(x)
    x = Conv2D(int(out_filters * r[Counter]), conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    Counter+=1
    return x,Counter

def resnet_block_fixed(input_data, in_filters, out_filters, conv_size, r):
    x = Conv2D(int(in_filters * r), conv_size, activation=None, padding='same')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(out_filters * r), conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    
    return x


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



def copy_weights(pre_trained_model, target_model, ranks_path):
    ranks = pd.read_csv(ranks_path, header=None).values

    rr = []
    for r in ranks:
        r = r[~np.isnan(r)]
        r = list(map(int, r))
        rr.append(r)

    i = 0
    last_filters = None  # Initialize last_filters
    
    for l_idx, l in enumerate(target_model.layers):
        if isinstance(l, Conv2D) or isinstance(l, Dense):
            if i == 0 and isinstance(l, Conv2D):
                i += 1
                continue  # Skip the first Conv2D layer
            
            conv_id = i - 1 if isinstance(l, Conv2D) else None
            if conv_id is not None and conv_id >= len(rr):
                print(f"Error: conv_id {conv_id} is out of range.")
                break
            
            if conv_id is not None:
                this_idcies = rr[conv_id][:l.filters]
                this_idcies = np.clip(this_idcies, 0, l.filters - 1)
                print(f"Conv layer {i}: {l.name}, this_idcies: {this_idcies}")
            else:
                this_idcies = None
            
            try:
                if isinstance(l, Conv2D):
                    pre_weights = pre_trained_model.layers[l_idx].get_weights()
                    if conv_id == 0:
                        weights = pre_weights[0][:, :, :, this_idcies]
                    else:
                        last_idcies = rr[conv_id - 1][:last_filters]
                        last_idcies = np.clip(last_idcies, 0, last_filters - 1)
                        weights = pre_weights[0][:, :, last_idcies, :][:, :, :, this_idcies]
                        
                        pad_width = l.filters - len(this_idcies)
                        if pad_width > 0:
                            weights = np.pad(weights, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant')

                    bias = pre_weights[1][this_idcies]
                    l.set_weights([weights, bias])
                    last_filters = l.filters  # Update last_filters
                    i += 1

                elif isinstance(l, Dense):
                    weights = pre_trained_model.layers[l_idx].get_weights()[0]
                    bias = pre_trained_model.layers[l_idx].get_weights()[1]
                    l.set_weights([weights, bias])

            except Exception as e:
                print(f"Error setting weights for layer {l.name}: {e}")
                continue
    
    return target_model
    
def main(opts):
    os.makedirs(opts.new_exp, exist_ok=True)

    # Setup params
    Batch_Size = 2048
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
    model_name = "resnet"
    pruning_start = datetime.now()
    if (opts.custom_pruning=='True'):
        r = np.loadtxt(opts.custom_pruning_file, delimiter=',')
        r = [1 - x for x in r]
        print("CUSTOM PRUNING RATE",r)

        print("---------------------------------------<<<<<<<<TARGET PRUNED ARCHITECTURE>>>>>>---------------------------------------")
        inp_shape = list(X_train.shape[1:])
        num_resnet_blocks = 5
        kernel_size = 5,1
        input_shp = list(X_train.shape[1:])
        rf_input = Input(shape=inp_shape, name = 'rf_input')
        Counter=0
        x = Conv2D(int(16*r[Counter]), (kernel_size), activation=None, padding='same')(rf_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        Counter+=1
        in_filters = int(16*r[Counter])
        out_filters = 32
        for i in range(num_resnet_blocks):
            if (i == num_resnet_blocks-1):
                out_filters = num_classes
            x,Counter = resnet_block_auto(x, in_filters, out_filters, kernel_size,r,Counter)
            in_filters = in_filters * 2
            out_filters = out_filters * 2

    else:

        r=(1-opts.pruning_rate)
        print("FIXED PRUNEIN AT ", r)

        print("---------------------------------------<<<<<<<<TARGET PRUNED ARCHITECTURE>>>>>>---------------------------------------")
        inp_shape = list(X_train.shape[1:])
        num_resnet_blocks = 5
        kernel_size = 5,1
        input_shp = list(X_train.shape[1:])
        rf_input = Input(shape=inp_shape, name = 'rf_input')

        x = Conv2D(int(16*r), (kernel_size), activation=None, padding='same')(rf_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        in_filters = int(16*r)
        out_filters = 32
        for i in range(num_resnet_blocks):
            if (i == num_resnet_blocks-1):
                out_filters = num_classes
            x = resnet_block_fixed(x, in_filters, out_filters, kernel_size,r)
            in_filters = in_filters * 2
            out_filters = out_filters * 2

    pruning_runtime = datetime.now() - pruning_start

    flatten = Flatten()(x)
    dropout_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(num_classes, activation='relu')(dropout_1)        
    softmax = Activation('softmax', name = 'softmax')(dense_1)

    optimizer= Adam(learning_rate=0.00060)
    model_pruned = keras.Model(rf_input, softmax)
    model_pruned.compile(loss='categorical_crossentropy', metrics=["accuracy"])

    print(model_pruned.summary())
    print("---------------------------------------<<<<<<<<PRETRAINED ARCHITECTURE>>>>>>---------------------------------------")

    model = load_model(opts.orignal_model)
    print("COPYING WEIGHTS FROM ORIGINAL MODEL")
    model.summary()
    
    model_pruned=copy_weights(model, model_pruned, opts.ranks_path)
    print("---------------------------------------<<<<<<<<TARGET ARCHITECTURE LOADED WITH WEIGHTS>>>>>>---------------------------------------")
    model_pruned.summary()
    
    
    model_pruned.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath =opts.new_exp + '/best_checkpoint.h5', 
                                                    verbose = 1,
                                                    save_best_only=True, 
                                                    save_weights_only=False,
                                                    mode='auto')
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            mode='auto',
            verbose = 1)


    start_finetune = datetime.now()

    # Train the model
    with tf.device('/GPU:0'):            
        history = model_pruned.fit(
            X_train,
            Y_train,
            batch_size=Batch_Size,
            epochs=150,
            verbose=1,
            validation_data=(X_val, Y_val   ),
            callbacks = [ cp_callback,earlystopping_callback]
            )

    finetune_runtime = datetime.now() - start_finetune

    best_checkpoint = opts.new_exp + '/best_checkpoint.h5'
    model_pruned.load_weights(best_checkpoint)
    # Show simple version of performance
    score = model_pruned.evaluate(X_test, Y_test,  verbose=0, batch_size=Batch_Size)
    print(score)
    # Plot confusion matrix
    test_Y_hat = model_pruned.predict(X_test, batch_size=Batch_Size)
    conf = np.zeros([num_classes,num_classes])
    confnorm = np.zeros([num_classes,num_classes])
    for i in range(0,X_test.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,num_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    saveplotpath=opts.new_exp+'/matrix.png'    
    plot_confusion_matrix(confnorm,saveplotpath, labels=classes)
    # Predict and calculate classification report
    Y_pred = model_pruned.predict(X_test, batch_size=Batch_Size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(Y_test, axis=1)
    classification_report_fp = classification_report(y_actual, y_pred, target_names=mods)

    # Print the classification report
    print(classification_report_fp)
    report_path = os.path.join(opts.new_exp, 'classification_report.txt')

    # Save the classification report to a file
    with open(report_path, 'w') as file:
        file.write(classification_report_fp)
    # Convert same_day_score to string
    same_day_score_str = str(score)

    # Write the string to the file
    with open(os.path.join(opts.new_exp, 'Accuracylos.txt'), "w") as file:
        file.write(same_day_score_str)

    print(type(history))
    print(type(history.history))

    print(f"Classification report saved to {report_path}")
    print(f"Pruning runtime: {pruning_runtime.total_seconds()}")
    print(f"Finetune runtime: {finetune_runtime.total_seconds()}")
    print(f"Finetune epochs: {len(history.history['loss'])}")
    print(f"Finetune runtime avg per epoch: {finetune_runtime.total_seconds()/len(history.history['loss'])}")





def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', help='')
    parser.add_argument('-o', '--new_exp', help='')
    parser.add_argument('-m', '--orignal_model', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='default value is 0')
    parser.add_argument('-pr', '--pruning_rate', type=float, default=0, help='')
    parser.add_argument('-rp', '--ranks_path', help='')
    parser.add_argument('-CP', '--custom_pruning',default='False', help='')
    parser.add_argument('-CPP', '--custom_pruning_file',default='False', help='')
    opts = parser.parse_args()
    return opts

    
if __name__ == "__main__":
    opts = parseArgs(sys.argv)

    main(opts)

