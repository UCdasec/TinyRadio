from pathlib import Path
import sys
import numpy as np
sys.path.append('models')
from typing import Dict, List, Tuple
import h5py
import pandas as pd
import os
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import shuffle
import h5py as h5
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path = Path('/workspace/Modulation/data/2021RadioML.hdf5')
name = 'radioml'
new_exp='/workspace/Modulation/experiments/ResNet/2021_27/BASELINE/'


# os.makedirs(new_exp, exist_ok=True)
os.makedirs(os.path.join(new_exp,'inference'), exist_ok=True)

def plot_confusion_matrix(cm, path, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure(figsize=(15,10))
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

def main():
    # Setup params
    Batch_Size = 1024
    emb_size = 64

    os.makedirs(os.path.join(new_exp), exist_ok=True)
    file_handle = h5.File(path, 'r+')

    myData = file_handle['X'][:]  #1024x2 samples 
    myMods = file_handle['Y'][:]  #mods 
    mySNRs = file_handle['Z'][:]  #snrs  

    file_handle.close()
   
    mods = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
            "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
            "FM", "GMSK", "OQPSK", "BFSK", "4FSK", "8FSK"]

    num_classes = 27

    # Split the data into train, validation, and test sets
    X_train, X_temp, Y_train, Y_temp, Z_train, Z_temp = train_test_split(
        myData, myMods, mySNRs, test_size=0.2, random_state=0)

    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(
        X_temp, Y_temp, Z_temp, test_size=0.5, random_state=0) 

    del myData, myMods, mySNRs
    best_checkpoint = new_exp + '/best_checkpoint.h5'

    # Load your model here
    model = keras.models.load_model(best_checkpoint)
    
  
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=Batch_Size)
    print(score)

    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=Batch_Size)
    conf = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(test_Y_hat, axis=1))
    confnorm = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    saveplotpath = os.path.join(new_exp,'inference', 'matrix.png')    
    plot_confusion_matrix(confnorm, saveplotpath, labels=mods)

    # Predict and calculate classification report
    Y_pred = model.predict(X_test, batch_size=Batch_Size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(Y_test, axis=1)
    classification_report_fp = classification_report(y_actual, y_pred, target_names=mods)
    print(classification_report_fp)

    report_path = os.path.join(new_exp, 'inference','classification_report.txt')
    with open(report_path, 'w') as file:
        file.write(classification_report_fp)
    print(f"Classification report saved to {report_path}")

    # Generate ROC curve
    fpr = dict()
    tpr = dict()
    thresholds = dict()  # Store thresholds
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test[:, i], Y_pred[:, i])  # Store thresholds here
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Save ROC curve data
    roc_data_path = os.path.join(new_exp, 'inference', 'roc_data.npz')
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)


    # Plot ROC curve
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {mods[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(new_exp,'inference', 'ROC_curve.png'))
    plt.show()

if __name__ == "__main__":
    main()
