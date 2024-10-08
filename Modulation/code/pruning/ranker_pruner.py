"""
Author: Ryan 

Combine the rank_gen.py and automatic_pruner.py

Useful for timing the whole pruning process instead of rank_gen time + 
automatic pruner.py 

BUGS: 
    - H =16,32,64 that require manaul intervention to score file
        - manual intervention means one of the files needs editing before 
            finetuning... so the automation is broken. use rank_gen and 
            automatic_pruner seperately instead.
"""


# Import necessary modules and packages
from enum import Enum
import matplotlib.pyplot as plt
import h5py as h5
import os
from warnings import warn
import pathlib
import sys
from scipy.spatial import distance  # Import distance calculation from SciPy
import argparse  # Import argument parsing library
from typing_extensions import Annotated
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    MaxPooling2D,
    Input,
    Conv2DTranspose,
    Concatenate,
    BatchNormalization,
    UpSampling2D,
    Dropout,
    Activation,
    GlobalAveragePooling1D,
    ZeroPadding2D,
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Flatten,
    Add,
)
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.activations import relu
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model  # Import TensorFlow and Keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mutual_info_score as MI,
)  # Import mutual information score from scikit-learn
import json
import re

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path

from datetime import datetime


console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


# Define a function to calculate mutual information between two arrays
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = MI(None, None, contingency=c_xy)
    return mi


# Define a function for quantization and mapping of weights
def mapping(W, min_w, max_w):
    scale_w = (max_w - min_w) / 100
    min_arr = np.full(W.shape, min_w)
    q_w = np.round((W - min_arr) / scale_w).astype(np.uint8)
    return q_w


# Define a function to rank feature maps in groups
def grouped_rank(feature_map, num_groups):
    dis = 256 / num_groups
    grouped_feature = np.round(feature_map / dis)
    r = np.linalg.matrix_rank(grouped_feature)
    return r


# Define a function to update distances between layers
def update_dis(Distances, layer_idx, dis):
    if layer_idx in Distances.keys():
        for k, v in dis.items():
            Distances[layer_idx][k] += v
    else:
        Distances[layer_idx] = dis
    return Distances


# Define a function to extract layers from a model
def extract_layers(model):
    layers = model.layers
    o = []
    model.summary()
    for i, l in enumerate(layers):
        if isinstance(l, Conv1D):
            o.append(l.output)
    return o


# Define a function to calculate rank of filters in each layer
def cal_rank(features, Results):
    for layer_idx, feature_layer in enumerate(features):
        after = np.squeeze(feature_layer)
        n_filters = after.shape[-1]
        filter_rank = list()
        if len(after.shape) == 2:
            for i in range(n_filters):
                a = after[:, i]
                rtf = np.average(a)
                filter_rank.append(rtf)
            filter_rank = sorted(filter_rank, reverse=True)
        else:
            filter_rank = sorted(after, reverse=True)
        filter_rank = mapping(
            np.array(filter_rank), np.min(filter_rank), np.max(filter_rank)
        )
        Results[layer_idx] = np.add(Results[layer_idx], np.array(filter_rank))
    return Results

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



# Define a function to extract feature maps from a model
def extract_feature_maps(opts, model, output_layers):
    dpath = opts.input
    tmp = opts.attack_window.split("_")
    attack_window = [int(tmp[0]), int(tmp[1])]
    method = opts.preprocess
    test_num = opts.max_trace_num
    Results = list()
    num_trace = 50
    extractor = Model(inputs=model.inputs, outputs=output_layers)
    for l in output_layers:
        Results.append(np.zeros(l.shape[-1]))
    whole_pack = np.load(dpath)
    x_data, plaintext, key = loadData.load_data_base(
        whole_pack, attack_window, method, test_num
    )
    for f in x_data[:num_trace]:
        x = np.expand_dims(f, axis=0)
        features = extractor(x)
        Results = cal_rank(features, Results)
    R_after = np.array(Results) / num_trace
    R_list = [list(r) for r in R_after]
    df = pd.DataFrame(R_list)
    df.to_csv(cur_path + "/stm_cnn_act.csv", header=False)
    return R_list


# Define a function for quantization and mapping of weights
def mapping(W, min_w, max_w):
    scale_w = (max_w - min_w) / 100
    min_arr = np.full(W.shape, min_w)
    q_w = np.round((W - min_arr) / scale_w).astype(np.uint8)
    return q_w


# Define a function to extract weights from a model
def extract_weights(model, output: Path):
    layers = model.layers[:-1]  # Skip the last layer
    model.summary()
    Results = list()
    idx_results = list()

    for l in layers:
        if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            if "classification" in l.name:
                continue

            a = l.get_weights()[0]
            if a.ndim == 4:  # Conv2D layer
                # Reshape the weights to (filters, -1)
                a = np.reshape(a, (a.shape[-1], -1))
            elif a.ndim == 2:  # Dense layer
                # Reshape the weights to (units, -1)
                a = np.reshape(a, (a.shape[1], -1))

            n_filters = a.shape[0]

            r = np.linalg.norm(a, axis=1)
            r = mapping(np.array(r), np.min(r), np.max(r))
            Results.append(sorted(r, reverse=True))
            idx_dis = np.argsort(r, axis=0)
            idx_results.append(idx_dis)

    # Save the results
    output.mkdir(exist_ok=True)
    df = pd.DataFrame(Results, index=None)
    df.to_csv(output.joinpath("l2.csv"), header=False, index=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(output.joinpath("l2_idx.csv"), header=False, index=False)

    return


# Define a function to apply FPGM (Filter Pruning via Geometric Median) on a model
def fpgm(model, opts, dist_type="l2"):
    layers = model.layers[:-1]  # Skip the last layer
    results = list()
    idx_results = list()
    r = list()

    for l in layers:
        if isinstance(l, (tf.keras.layers.Conv1D, tf.keras.layers.Dense)):
            print(l.name)
            w = l.get_weights()[0]
            weight_vec = np.reshape(w, (-1, w.shape[-1]))

            if dist_type == "l2" or dist_type == "l1":
                dist_matrix = distance.cdist(
                    np.transpose(weight_vec), np.transpose(weight_vec), "euclidean"
                )
            elif dist_type == "cos":
                dist_matrix = 1 - distance.cdist(
                    np.transpose(weight_vec), np.transpose(weight_vec), "cosine"
                )

            squeeze_matrix = np.sum(np.abs(dist_matrix), axis=0)
            distance_sum = sorted(squeeze_matrix, reverse=True)
            idx_dis = np.argsort(squeeze_matrix, axis=0)
            r = mapping(
                np.array(distance_sum), np.min(distance_sum), np.max(distance_sum)
            )
            results.append(r)
            idx_results.append(idx_dis)
            r = list()

    os.makedirs(opts.output, exist_ok=True)
    df = pd.DataFrame(results, index=None)
    df.to_csv(os.path.join(opts.output, "fpgm.csv"), header=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(os.path.join(opts.output, "fpgm_idx.csv"), header=False)


# Parse command line arguments
# def parseArgs(argv):
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-o', '--output', help='')
#    parser.add_argument('-i', '--model_dir', help='')
#    parser.add_argument('-type', '--type', choices={'l2', 'fpgm'}, help='')
#    opts = parser.parse_args()
#    return opts


class PruneType(str, Enum):
    l2 = "l2"
    fpgm = "fpgm"


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
                this_idcies = rr[conv_id][: l.filters]
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
                        weights = pre_weights[0][:, :, last_idcies, :][
                            :, :, :, this_idcies
                        ]

                        pad_width = l.filters - len(this_idcies)
                        if pad_width > 0:
                            weights = np.pad(
                                weights,
                                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                                mode="constant",
                            )

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


def load_radio_mod_data(dataset: Path):
    """
    Load the radio mode data
    """

    file_handle = h5.File(dataset, "r+")

    new_myData = file_handle["X"][:]  # 1024x2 samples
    new_myMods = file_handle["Y"][:]  # mods
    new_mySNRs = file_handle["Z"][:]  # snrs

    file_handle.close()
    myData = []
    myMods = []
    mySNRs = []
    # Define the threshold
    threshold = 6
    for i in range(len(new_mySNRs)):
        if new_mySNRs[i] >= threshold:
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
    # First split: 80% train, 20% temp (test + validation)
    X_train, X_temp, Y_train, Y_temp, Z_train, Z_temp = train_test_split(
        myData, myMods, mySNRs, test_size=0.2, random_state=0
    )

    # Second split: 50% of the temp data for validation, 50% for testing (since it's 10% of the original data)
    X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(
        X_temp, Y_temp, Z_temp, test_size=0.5, random_state=0
    )

    del myData, myMods, mySNRs

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def resnet_block_auto(input_data, in_filters, out_filters, conv_size, r, Counter):
    print(r[Counter])
    x = Conv2D(
        int(in_filters * r[Counter]), conv_size, activation=None, padding="same"
    )(input_data)
    x = BatchNormalization()(x)
    Counter += 1
    # x = Add()([x, input_data])
    print(r[Counter])
    x = Activation("relu")(x)
    x = Conv2D(
        int(out_filters * r[Counter]), conv_size, activation=None, padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="same")(x)
    Counter += 1
    return x, Counter


def resnet_block_fixed(input_data, in_filters, out_filters, conv_size, r):
    x = Conv2D(int(in_filters * r), conv_size, activation=None, padding="same")(
        input_data
    )
    x = BatchNormalization()(x)
    # x = Add()([x, input_data])
    x = Activation("relu")(x)
    x = Conv2D(int(out_filters * r), conv_size, activation=None, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="same")(x)

    return x


# TODO: Difference between providing l2.csv and l2_idx.csv
# IMPORTANT!!


def custom_prune_model(
    model_path: Path, custom_pruning_file: Path, ranks_path: Path, X_train, Y_train
):
    """
    Prune the model
    """

    if "_idx.csv" not in ranks_path.name:
        warn(f"Ranks path is the l2_idx.csv file")
        warn(f"In the bash hist file... l2_idx.csv is passwas as -rp parameter")
        warn(f"while... l2.csv is passed as -i parameter")
        warn(f"both are refered to as ranks_path in the code but used differently:(")
        warn(
            f" arg -rp (l2_idx.csv) is for automatic_training, while -i (l2.csv) is for automatic pruning"
        )

        # In automatic training the l2_idx is used for the copy weights functions
        # in automatic pruning the l2.csv file is used gratitude pruning

        raise Exception("Ranks path must be the l2_idx.csv file!... i think")

    warn(f"Looks like 'automatic_pruner.py' takes l2.csv based on mabons HIST")
    warn(f"Looks like 'automatic_training.py' takes l2_idx.csv based on mabons HIST")

    # TODO: Better to deduce this from data... incase we one day get a new
    # mod dataset with more
    num_classes = 27

    inp_shape = list(X_train.shape[1:])

    r = np.loadtxt(custom_pruning_file, delimiter=",")
    r = [1 - x for x in r]

    inp_shape = list(X_train.shape[1:])
    num_resnet_blocks = 5
    kernel_size = 5, 1

    rf_input = Input(shape=inp_shape, name="rf_input")
    Counter = 0
    x = Conv2D(int(16 * r[Counter]), (kernel_size), activation=None, padding="same")(
        rf_input
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    Counter += 1
    in_filters = int(16 * r[Counter])
    out_filters = 32
    for i in range(num_resnet_blocks):
        if i == num_resnet_blocks - 1:
            out_filters = num_classes
        x, Counter = resnet_block_auto(
            x, in_filters, out_filters, kernel_size, r, Counter
        )
        in_filters = in_filters * 2
        out_filters = out_filters * 2

    flatten = Flatten()(x)
    dropout_1 = Dropout(0.5)(flatten)
    dense_1 = Dense(num_classes, activation="relu")(dropout_1)
    softmax = Activation("softmax", name="softmax")(dense_1)

    model_pruned = keras.Model(rf_input, softmax)
    model_pruned.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    # Load the original model and copy the weights
    model = load_model(model_path)
    model_pruned = copy_weights(model, model_pruned, ranks_path)

    return model_pruned


def finetune_model(
    model, x_train, y_train, x_val, y_val, checkpoint_dir: Path, batch_size: int
):
    """
    Finetune the model
    """

    best_checkpoint = checkpoint_dir.joinpath("pruned_best_checkpoint.h5")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="auto", verbose=1
    )

    # Train the model
    with tf.device("/GPU:0"):
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=150,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[cp_callback, earlystopping_callback],
        )

    model.load_weights(best_checkpoint)

    return model, history


def test_model(model, X_test, Y_test, Batch_Size, out: Path):
    # Show simple version of performance


    start = datetime.now()
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=Batch_Size)
    runtime = datetime.now() - start
    print(score)

    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=Batch_Size)

    mods = [
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
        "BFSK",
        "4FSK",
        "8FSK",
    ]

    num_classes = 27

    conf = np.zeros([num_classes, num_classes])
    confnorm = np.zeros([num_classes, num_classes])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, num_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    saveplotpath = out / "matrix.png"

    plot_confusion_matrix(confnorm, saveplotpath, labels=mods)

    # Predict and calculate classification report
    Y_pred = model.predict(X_test, batch_size=Batch_Size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(Y_test, axis=1)
    classification_report_fp = classification_report(
        y_actual, y_pred, target_names=mods
    )

    # Print the classification report
    print(classification_report_fp)
    report_path = out / "classification_report.txt"

    # Save the classification report to a file
    with open(report_path, "w") as file:
        file.write(classification_report_fp)
    # Convert same_day_score to string
    same_day_score_str = str(score)

    # Write the string to the file
    with open(out.joinpath("Accuracylos.txt"), "w") as file:
        file.write(same_day_score_str)

    # print(f"Classification report saved to {report_path}")
    # print(f"Pruning runtime: {pruning_runtime.total_seconds()}")
    # print(f"Finetune runtime: {finetune_runtime.total_seconds()}")
    # print(f"Finetune epochs: {len(history.history['loss'])}")
    # print(f"Finetune runtime avg per epoch: {finetune_runtime.total_seconds()/len(history.history['loss'])}")
    return runtime


# Define a function for gratitude-based pruning
def gratitude_pr(rank_result, n, out_dir: Path):
    gratitude = list()
    pruning_rate = list()
    idxs = list()
    for rank in rank_result:
        rank = rank[1:]
        gra = list()
        for idx, r in enumerate(rank):
            if idx == len(rank) - n:
                break
            try:
                g = (rank[idx + n] - r) / n  # Calculate gratitude
                gra.append(g)
            except IndexError:
                pass

        gratitude.append(np.array(gra))
    for gra in gratitude:
        for idx, g in enumerate(gra):
            if g == max(gra):
                idxs.append(
                    int(idx + n / 2)
                )  # Find the index with the maximum gratitude
                pruning_rate.append(
                    float("{:.2f}".format(1 - (int(idx + n / 2)) / len(gra)))
                )  # Calculate pruning rate
                break
    for i in range(len(pruning_rate)):
        if pruning_rate[i] > 0.9:
            pruning_rate[i] = 0.9
        elif pruning_rate[i] < 0:
            pruning_rate[i] = 0

    # Convert the list to a numpy array
    pruning_rate = np.array(pruning_rate)
    # Save the numpy array to a CSV file
    np.savetxt(out_dir.joinpath("1-pr.csv"), pruning_rate, delimiter=",")
    return


def automatic_pruner(rank_path: Path, out: Path, n: int):
    """
    Calling gratidute pr
    """

    rank_result = pd.read_csv(rank_path, header=None).values
    rr = list()
    for r in rank_result:
        r = r[~np.isnan(r)]
        rr.append(r)
    #os.makedirs(opts.output, exist_ok=True)

    # Call gratitude_pr function with the rank results and specified parameter N
    gratitude_pr(rr, n, out)

    return


@app.command()
def prune_automatic(
    model_path: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Argument()],
    prune_type: Annotated[PruneType, typer.Argument()],
    h_val: Annotated[int, typer.Argument()],
    show_model_summary: Annotated[bool, typer.Option()] = True,
    skip_finetune: Annotated[bool, typer.Option()] = False,
    dataset: Annotated[Path, typer.Argument()] = Path("~/TinyRadio/Modulation/data/2021RadioML.hdf5").expanduser(),
):
    """
    Generate the ranks and prune the model.
    Ranks and model's are saved to out direcrory
    """

    # 1. load the model  load the data
    model = load_model(model_path)
    if show_model_summary:
        model.summary()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_radio_mod_data(dataset)

    # 2. Generate the pruning ranks (rank_gen.py) generate l2.csv and l2_idx.csv
    tot_start = datetime.now()
    prune_start = datetime.now()
    extract_weights(model, out)

    # 3. Generate pruning file (automatic_pruner.py) -- takes l2.csv from rank gen
    #           as input. Outputs 1-pr.csv
    automatic_pruner(out.joinpath("l2.csv"), out, h_val)

    # 4. Apply the custom pruning rate to the model to create pruned model
    custom_pruning_file = out.joinpath("1-pr.csv")
    ranks_path = out.joinpath("l2_idx.csv")
    pruned_model = custom_prune_model(
        model_path, custom_pruning_file, ranks_path, X_train, Y_train
    )
    prune_runtime = datetime.now() - prune_start

    if show_model_summary: pruned_model.summary()

    if not skip_finetune:
        # 5. Finetune pruned model, + load data
        finetuned_pruned_model, history = finetune_model(
            pruned_model, X_train, Y_train, X_val, Y_val, out, batch_size=2048
        )


        # 6. Test the final model:D
        eval_time = test_model(finetuned_pruned_model, X_test, Y_test, 2048, out)

        print(f"Test took: {eval_time.total_seconds()}")
        print(f"Test had: {X_test.shape[0]} traces")
        print(f"Therefore: {X_test.shape[0]/eval_time.total_seconds()} traces per second")

    tot_runtime = datetime.now() - tot_start

    print(f"Prune runtime: {prune_runtime.total_seconds()} seconds")
    print(f"Tot Prune runtime: {tot_runtime.total_seconds()} seconds")

    return


@app.command()
def prune_fixed(
    model: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Argument()],
    # TODO: Do I need prune type when doing fixed?
    prune_type: Annotated[PruneType, typer.Argument()],
    prune_rate: Annotated[PruneType, typer.Argument()],
):
    """
    Generate the ranks and prune the model.
    Ranks and model's are saved to out direcrory
    """

    return


if __name__ == "__main__":
    app()

    # model = load_model(opts.model_dir)
    # model.summary()
    #
    # if opts.type == 'l2':
    #    start = datetime.now()
    #    extract_weights(model, opts)
    #    runtime = datetime.now() - start
    #    print(f"Total time was {runtime.total_seconds()}")
    # else:
    #    fpgm(model, opts)
