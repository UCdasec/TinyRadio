#!/usr/bin/python3
from __future__ import division
import tensorflow as tf
import tensorflow.keras as keras
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
import time
import load_slice_IQ
import config
#import get_simu_data
from tools import shuffleData
from pathlib import Path
import sys
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
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import  Dropout, Activation, GlobalAveragePooling1D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.layers import Reshape, Dense, Flatten, Add
from tensorflow.keras.activations import relu
from sklearn.metrics import roc_curve, auc

new_exp='/home/mabon/TinyRadio/RFP/experiments/ResNet/NEU/BASELINE/'
#new_exp = 'ryan_exp'
#n2_best ='/home/mabon/TinyRadio/RFP/experiments/ResNet/NEU/pruned/automatic/N2/best_checkpoint.h5'

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

class LearningController(Callback):
    def __init__(self, num_epoch=0, lr = 0., learn_minute=0):
        self.num_epoch = num_epoch
        self.learn_second = learn_minute * 60
        self.lr = lr
        if self.learn_second > 0:
            print("Leraning rate is controled by time.")
        elif self.num_epoch > 0:
            print("Leraning rate is controled by epoch.")

    def on_train_begin(self, logs=None):
        if self.learn_second > 0:
            self.start_time = time.time()
        self.model.optimizer.lr = self.lr


    def on_epoch_end(self, epoch, logs=None):
        if self.learn_second > 0:
            current_time = time.time()
            if current_time - self.start_time > self.learn_second:
                self.model.stop_training = True
                print("Time is up.")
                return

            if current_time - self.start_time > self.learn_second / 2:
                self.model.optimizer.lr = self.lr * 0.1
            if current_time - self.start_time > self.learn_second * 3 / 4:
                self.model.optimizer.lr = self.lr * 0.01

        elif self.num_epoch > 0:
            if epoch >= self.num_epoch / 3:
                self.model.optimizer.lr = self.lr * 0.1
            if epoch >= self.num_epoch * 2 / 3:
                self.model.optimizer.lr = self.lr * 0.01

        print('lr:%.2e' % self.model.optimizer.lr.value())

def plot_and_save_results(X, y, model, num_classes, test_type, acc_list,Batch_Size):
    # Plot confusion matrix
    test_Y_hat = model.predict(X, batch_size=Batch_Size)
    conf = np.zeros([num_classes, num_classes])
    confnorm = np.zeros([num_classes, num_classes])
    for i in range(0, X.shape[0]):
        j = list(y[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] += 1
    for i in range(0, num_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    saveplotpath = new_exp + f'/{test_type}matrix.png'
    classes_lab = [f"Device {i + 1}" for i in range(num_classes)]
    plot_confusion_matrix(confnorm, saveplotpath, labels=classes_lab)

    # Predict and calculate classification report
    Y_pred = model.predict(X, batch_size=Batch_Size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(y, axis=1)
    classification_report_fp = classification_report(y_actual, y_pred, target_names=classes_lab)

    # Print the classification report
    print(classification_report_fp)
    report_path = os.path.join(new_exp, f'{test_type}classification_report.txt')

    # Save the classification report to a file
    with open(report_path, 'w') as file:
        file.write(classification_report_fp)

    print(f"Classification report saved to {report_path}")

    # Generate and save ROC curve
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    roc_data_path = os.path.join(new_exp, f'{test_type}roc_data.npz')
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)

    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes_lab[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(new_exp, f'{test_type}ROC_curve.png'))
    plt.close()


def resnet_block(input_data, in_filters, out_filters, conv_size):
  x = Conv2D(in_filters, conv_size, activation=None, padding='same')(input_data)
  x = BatchNormalization()(x)
#   x = Add()([x, input_data])
  x = Activation('relu')(x)
  x = Conv2D(out_filters, conv_size, activation=None, padding='same')(x) 
  x = BatchNormalization()(x)
  x = MaxPooling2D(2, strides = (2,1), padding = 'same') (x)
  return x



def main(opts):
    # load data
    same_acc_list = []
    cross_acc_list = []

    # setup params
    Batch_Size = 1024
    Epoch_Num = 150
    lr = 0.1
    emb_size = 64
    idx_list = [0]
    # idx_list = [0,0,0]
    for idx in idx_list:
        dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, num_slice=opts.num_slice,
                                              slice_len=opts.slice_len, start_idx=idx, stride=opts.stride,
                                              mul_trans=opts.mul_trans, window=opts.window, dataType=opts.dataType)

        train_x, train_y, val_x, val_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)

        num_classes=20        
        print('get the model and compile it...')
        print (np.shape(train_x))
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1, train_x.shape[2])
        val_x  = val_x.reshape(val_x.shape[0], val_x.shape[1], 1, val_x.shape[2])
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], 1, test_x.shape[2])


        print (np.shape(train_x))

        num_resnet_blocks = 5
        kernel_size = 5,1
        input_shp = list(train_x.shape[1:])
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

        optimizer= Adam(learning_rate=0.01)
        model = keras.Model(rf_input, softmax)
        model.compile(loss='categorical_crossentropy', metrics=["accuracy"])

        print(model.summary())

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=new_exp + '/best_checkpoint.h5', 
                                                         verbose=1, save_best_only=True, save_weights_only=False,
                                                         mode='auto')
        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='auto', verbose=1)

        train_y = to_categorical(train_y, num_classes=NUM_CLASS)
        val_y = to_categorical(val_y, num_classes=NUM_CLASS)
        test_y = to_categorical(test_y, num_classes=NUM_CLASS)

        num_classes = NUM_CLASS

        start = datetime.now()
        # Train the model
        #with tf.device('/GPU:0'):
        #    history = model.fit(train_x, train_y, batch_size=Batch_Size, epochs=Epoch_Num, verbose=1,
        #                        validation_data=(val_x, val_y), callbacks=[cp_callback, earlystopping_callback])

        runtime = datetime.now() - start
        print(f"[RYAN HERE] runtime was {runtime.total_seconds()} seconds")
        print(f"[RYAN HERE] epochs was {len(history.history['loss'])} epochs")
        print(f"[RYAN HERE] time per epochs was {runtime.total_seconds()/len(history.history['loss'])} sec per epoch")
#
        best_checkpoint = new_exp + '/best_checkpoint.h5'

        model.load_weights(best_checkpoint)
        #model = tf.keras.models.load_model(n2_best)

        # Evaluate on same day test set
        print("SAMEDAY TEST")
        start = datetime.now()
        same_day_score = model.evaluate(test_x, test_y, verbose=1, batch_size=Batch_Size)
        runtime = datetime.now() - start 
        print(same_day_score)
        print(f"Runtime: {runtime.total_seconds()} seconds")
        print(f"Frame: {len(test_x)}")
        print(f"FPS: {len(test_x)/runtime.total_seconds()} frames per seconds")
        plot_and_save_results(test_x, test_y, model, num_classes, 'SAMEDAY', same_acc_list,Batch_Size)
        # Convert same_day_score to string
        same_day_score_str = str(same_day_score)

        # Write the string to the file
        with open(os.path.join(new_exp, 'Accuracylos.txt'), "w") as file:
            file.write(same_day_score_str)

        # # Evaluate on cross day test set
        # print("start testing on cross day scenario...")
        # cross_day_dataOpts = load_slice_IQ.loadDataOpts(opts.testData, opts.location, num_slice=opts.num_slice,
        #                                                 slice_len=opts.slice_len, start_idx=idx, stride=opts.stride,
        #                                                 mul_trans=opts.mul_trans, window=opts.window, dataType=opts.dataType)
        # cross_day_dataOpts.num_slice = int(opts.num_slice * 0.2)
        # X, y, _, _, _, _, NUM_CLASS = load_slice_IQ.loadData(cross_day_dataOpts, split=False)
        # y = to_categorical(y, NUM_CLASS)
        # X, y = shuffleData(X, y)

        # cross_day_score, cross_day_acc = model.evaluate(X, y, batch_size=Batch_Size, verbose=1)
        # print('Cross-Day test accuracy is:', cross_day_acc)
        # cross_acc_list.append(cross_day_acc)
        # plot_and_save_results(X, y, model, NUM_CLASS, 'CROSSDAY', cross_acc_list,Batch_Size)



class testOpts():
    
    def __init__(self, trainData, testData, location, modelType, num_slice, slice_len, start_idx, stride, window, dataType):
        self.input = trainData
        self.testData = testData
        self.modelType = modelType
        self.location = location
        self.verbose = 1
        self.trainData = trainData
        self.splitType = 'random'
        self.normalize = False
        self.dataSource = 'neu'
        self.num_slice = num_slice
        self.slice_len = slice_len
        self.start_idx = start_idx
        self.stride = stride
        self.window = window
        self.mul_trans = True
        self.dataType = dataType


if __name__ == "__main__":
    # opts = config.parse_args(sys.argv)
    just_read_data = True


    #TODO: our_day4 is what all the code runs on 
    #source = ['our_day4']

    source = ['our_day4']
    target = ['our_day2']
    data = list(zip(source, target))
    for s in [864]:
        for w in [64]:
            for m in ['resnet']:     
                for p in data:
                    dataPath = '/home/mabon/NEU/' + p[0]
                    #dataPath = '/home/mabon/TinyRadio/RFP/data/NEU/' + p[0]
                    testPath = '/home/mabon/TinyRadio/RFP/data/NEU/' + p[1]
                    opts = testOpts(trainData=dataPath, testData=testPath, location='after_equ', modelType= m, num_slice= 40000, slice_len= 864, start_idx=0, stride = s, window=w, dataType='IQ')
                    if not just_read_data:
                        main(opts)
                    else:
                        # setup params
                        Batch_Size = 1024
                        Epoch_Num = 150
                        lr = 0.1
                        emb_size = 64
                        idx_list = [0]
                        # idx_list = [0,0,0]

                        for idx in idx_list:
                            dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, num_slice=opts.num_slice,
                                                                  slice_len=opts.slice_len, start_idx=idx, stride=opts.stride,
                                                                  mul_trans=opts.mul_trans, window=opts.window, dataType=opts.dataType)

                            train_x, train_y, val_x, val_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)
                            print(f"THIS IS FROM DAY {source}")

