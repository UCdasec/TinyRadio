"""
Author: Ryan 

Combine the rank_gen.py and automatic_pruner.py

Useful for timing the whole pruning process instead of rank_gen time + 
automatic pruner.py 

BUGS: 
    - With h =16,32,64 that require manaul  intervention 
        - manual intervention means one of the files needs editing before 
            finetuning... so this script will not work when h>8. use rank_gen and 
            automatic_pruner seperately instead and manually edit the files
"""

import load_slice_IQ
from tensorflow.keras.utils import to_categorical
from enum import Enum
import h5py as h5
import os
from warnings import warn
from pathlib import Path
from scipy.spatial import distance
from typing_extensions import Annotated
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    MaxPooling2D,
    Input,
    BatchNormalization,
    Dropout,
    Activation,
    Dense,
    Flatten,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mutual_info_score as MI,
)  
import typer
from rich.console import Console
from pathlib import Path

from datetime import datetime
import matplotlib.pyplot as plt


console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


def plot_confusion_matrix(cm,path:Path, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    """
    Plot the conf matrix
    """
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
    plt.savefig(path)


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
    num_classes = 20

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


    #TODO: num classes hardcoded 
    y_train = to_categorical(y_train, num_classes=20)
    y_val = to_categorical(y_val, num_classes=20)

    # Train the model with train dataset
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


def test_model(model, X_test, Y_test, batch_size, out: Path):

    #TODO: num classes hardcoded 
    Y_test = to_categorical(Y_test, num_classes=20)

    # Show simple version of performance
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    print(f"[SCORE] {score}")

    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)

    devices = [f'Device_{i}' for i in range(20)]

    num_classes = 20

    conf = np.zeros([num_classes, num_classes])
    confnorm = np.zeros([num_classes, num_classes])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, num_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    saveplotpath = out / "matrix.png"

    plot_confusion_matrix(confnorm, saveplotpath, labels=devices)

    # Predict and calculate classification report
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_actual = np.argmax(Y_test, axis=1)
    classification_report_fp = classification_report(
        y_actual, y_pred, target_names=devices
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

    return


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



def load_neu():
    """
    """

    #COPED FROM 'automatic_pruner.py'

    source = ['our_day4']
    target = ['our_day2']
    s = 864
    w = 64
    p = list(zip(source, target))[0]
    m = "resnet"

    #TODO: TEST PATH IS NEVER USED!
    dataPath = '/home/mabon/NEU/' + p[0] 

    testPath = '/home/mabon/TinyRadio/RFP/data/HackRF10_dataset_cns22/' + p[1]
    opts = testOpts(trainData=dataPath, testData=testPath, location='after_equ', modelType= m, num_slice= 40000, slice_len= 864, start_idx=0, stride = s, window=w, dataType='IQ')

    # load data
    same_acc_list = []
    #cross_acc_list = []

    # setup params
    Batch_Size = 1024
    Epoch_Num = 150
    lr = 0.1
    emb_size = 64
    idx = 0 

    dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, num_slice=opts.num_slice,
                                          slice_len=opts.slice_len, start_idx=idx, stride=opts.stride,
                                          mul_trans=opts.mul_trans, window=opts.window, dataType=opts.dataType)

    train_x, train_y, val_x, val_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1, train_x.shape[2])
    val_x  = val_x.reshape(val_x.shape[0], val_x.shape[1], 1, val_x.shape[2])
    test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], 1, test_x.shape[2])

    return train_x, train_y, val_x, val_y, test_x, test_y


@app.command()
def prune_automatic(
    model_path: Annotated[Path, typer.Argument()],
    out: Annotated[Path, typer.Argument()],
    prune_type: Annotated[PruneType, typer.Argument()],
    h_val: Annotated[int, typer.Argument()],
    show_model_summary: Annotated[bool, typer.Option()] = True,
    override_pr_files: Annotated[Path, typer.Option(help="Load previously generated scores but still time")] = None,
    skip_finetune: Annotated[bool, typer.Option()] = False,
    dataset: Annotated[Path, typer.Argument()] = Path("~/TinyRadio/Modulation/data/2021RadioML.hdf5").expanduser(),
    verbose: Annotated[bool, typer.Option()] = False,
    stop_after_load_data: Annotated[bool, typer.Option()] = False,
):
    """
    Generate the ranks and prune the model.
    Ranks and model's are saved to out direcrory
    """

    # For NEU, batch size has been set to 1024
    # NOTICE: Radio mod uses 2048... don't see this being issue just notice
    #  test size for this is much smaller, 4000, so I woold choose even less 
    #   than 1024
    #TODO: Paper says test size is 4000 traces, I get something different 
    batch_size = 1024

    # 1. load the model  load the data
    model = load_model(model_path)
    if show_model_summary:
        model.summary()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_neu()

    if verbose or stop_after_load_data:
        tot = X_train.shape[0] + X_val.shape[0] + X_test[0]
        print(f"Total IQ traces {tot}")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
    if stop_after_load_data:
        return



    # 2. Generate the pruning ranks (rank_gen.py) generate l2.csv and l2_idx.csv
    prune_start = datetime.now()
    if override_pr_files is not None:
        tmp_out = out.joinpath("NOT_USD")
        tmp_out.mkdir()
        extract_weights(model, tmp_out)
    else:
        extract_weights(model, out)

    # 3. Generate pruning file (automatic_pruner.py) -- takes l2.csv from rank gen
    #           as input. Outputs 1-pr.csv
    if override_pr_files is not None:
        # Use the provided l2.csv as input, output some other 1-pr
        automatic_pruner(out.joinpath("l2.csv"), tmp_out, h_val)
    else:
        automatic_pruner(out.joinpath("l2.csv"), out, h_val)

    # 4. Apply the custom pruning rate to the model to create pruned model
    custom_pruning_file = out.joinpath("1-pr.csv")
    ranks_path = out.joinpath("l2_idx.csv")

    pruned_model = custom_prune_model(
        model_path, custom_pruning_file, ranks_path, X_train, Y_train
    )
    prune_runtime = datetime.now() - prune_start
    
    if verbose:
        print(f"Prune runtime was: {prune_runtime.total_seconds()}")

    if show_model_summary: pruned_model.summary()

    if not skip_finetune:
        # 5. Finetune pruned model, + load data
        finetuned_pruned_model, history = finetune_model(
            pruned_model, X_train, Y_train, X_val, Y_val, out, batch_size=batch_size
        )

        # 6. Test the final model:D
        test_model(finetuned_pruned_model, X_test, Y_test, batch_size=batch_size, out=out)

    print(f"Prune runtime: {prune_runtime.total_seconds()} seconds")

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
    Ranks and model's are saved to out direcrory.
    """



    return



def accumulate_ranks(preds, num_class, target):
    """
    """

    # Get the total number of traces
    trace_num = preds.shape[0]
    rank = []

    #  Init blank scores
    scores = np.zeros(num_class)

    # Iterate over all the traces in the prediction 
    # For each trace 
    for i in range(trace_num):

        # Scores is an element wise summation of the 
        # prediction 
        scores += preds[i]

        # Indices that would sort the array, flipped. 
        # The indice of each prediction is the predicted target 

        # Therefore, a list of indices that would sort the array (lowest to high) is the (least_predic_device to the most_predicted_device).
        # Therefore... fliiping the list gives us the 
        # (most_predicted_device to least_predicted_device)
        r = np.argsort(scores)[::-1]

        # Get the indice of the target device in our list, the 
        # [0][0] gets the probability of the prediction 
        rank.append(np.where(r==target)[0][0])

    return rank


def class_ranks(model, testX, testY, num_class, preds=None):
    ranks = []

    testY = np.array(testY)

    print(f"text x shape: {testX.shape}")
    for i in range(num_class):

        # index of where device label = i 
        ids = np.where(testY == i)

        if preds is None:

            # Pull all traces where label was 1 
            OneClsData = np.take(testX, ids, axis=0)
            print(f" ONE CLASS DATA shape: {OneClsData.shape}")

            # Squeeze the traces... dunno why tbh  
            OneClsData = np.squeeze(OneClsData, axis=0)
            print(f" ONE CLASS DATA shape post squeeze: {OneClsData.shape}")

            # Predict give the all the traces for device i 
            OneClsPreds = model.predict(OneClsData, verbose=1)
        else:
            OneClsPreds = np.take(preds, ids, axis=0)

        OneCls_rank = accumulate_ranks(np.squeeze(OneClsPreds), num_class, i)

        ranks.append(OneCls_rank)
    return ranks


@app.command()
def rank_plot(
    model_path_base: Annotated[Path, typer.Argument(help='model checkpoint path')],
    model_path_h2: Annotated[Path, typer.Argument(help='model checkpoint path')],
    model_path_h4: Annotated[Path, typer.Argument(help='model checkpoint path')],
    #model_path_h8: Annotated[Path, typer.Argument(help='model checkpoint path')],
    plot_path: Annotated[Path, typer.Argument(help='Path to save plot')],
    use_cache: Annotated[bool, typer.Option(help='Use existing rank.csv')]=False,
    ):
    """
    Generate the rank plots 

    Notice: This will plot the rank for all the devices, _not_ the average 
        device rank 
    """

    if not use_cache:
        _, _, _, _, X_test, Y_test = load_neu()


        # Load the ranks, return array has <NUM_DEVICE> rows and <NUM_TRACES> 
        # columns. 


        model = load_model(model_path_base)
        ranks_base = np.array(class_ranks(model, X_test, Y_test, 20, preds=None))

        model = load_model(model_path_h2)
        ranks_h2 = np.array(class_ranks(model, X_test, Y_test, 20, preds=None))

        model = load_model(model_path_h4)
        ranks_h4 = np.array(class_ranks(model, X_test, Y_test, 20, preds=None))
        #model = load_model(model_path_h8)
        #ranks_h8 = np.array(class_ranks(model, X_test, Y_test, 20, preds=None))

        # Save arrays 
        np.save("rank_result/sameday_rank_base.npy", ranks_base)
        np.save("rank_result/sameday_rank_h2.npy", ranks_h2)
        np.save("rank_result/sameday_rank_h4.npy", ranks_h4)
        #np.save("rank_result/sameday_rank_h8.npy", ranks_h8)
    else:
        #df = pd.read_csv("rank_result/sameday_rank_RYAN.csv")#,header=False)
        ranks_base= np.load("rank_result/sameday_rank_base.npy")
        ranks_h2 = np.load("rank_result/sameday_rank_h2.npy")
        ranks_h4 = np.load("rank_result/sameday_rank_h4.npy")
        #ranks_h8 = np.load("rank_result/sameday_rank_h8.npy")


    # CSV is    rank trace 1, rank trace 2, rank trace 3.....
    # device 1 
    # device2 
   #  device3
    # ...
    # ...
    # now get the mean rank per device
    mean_base= ranks_base.mean(axis=0)
    mean_h2= ranks_h2.mean(axis=0)
    mean_h4= ranks_h4.mean(axis=0)

    data = {
        'baseline': mean_base,
        'h2' : mean_h2,
        'h4' : mean_h4,
        #'h8' : mean_h8,
    }

    fig = plt.plot(data['baseline'], linewidth=2,label='Baseline' ,marker='^',markevery=20,markersize=10,color='blue')
    plt.plot(data['h4'], linewidth=2,label='Pruned (h=4)' ,marker='^',markevery=20,markersize=10,color='red')
    plt.plot(data['h2'], linewidth=2,label='Pruned (h=2)' ,marker='^',markevery=20,markersize=10, color='green')

    plt.xlim(0,50)
    plt.yticks([i for i in range(0,20,2)],fontsize=20)
    plt.xticks(fontsize=20)
    #plt.legend(title='Device')
    plt.legend(fontsize=20)
    plt.xlabel('No. of Test Traces', fontsize=20)
    plt.ylabel('Device Rank', fontsize=20)
    plt.subplots_adjust(bottom=0.15)  
    plt.show()

    plt.savefig(plot_path)

    return 


if __name__ == "__main__":
    app()
