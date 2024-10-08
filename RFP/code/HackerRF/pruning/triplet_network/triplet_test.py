#! /usr/bin/env python
from __future__ import division
import os
import sys
import argparse
import pdb
import time
from statistics import mean, stdev
import random
import time
import numpy as np

from tensorflow.keras.models import load_model

RootDir = os.getcwd()
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(toolsDir)
import load_slice_IQ
from utility import create_test_set_Wang_disjoint, kNN_accuracy, getRFdataDict


currentDir = os.path.dirname(__file__)
ResDir = os.path.join(currentDir, 'TF/res_out')
os.makedirs(ResDir, exist_ok=True)


def Wang_Disjoint_Experment(opts, modelPath, n_shot, max_n = 20):
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with different distributions.
    The model is trained on AWF777 and tested on Wang100 and the set of
    websites in the training set and the testing set are mutually exclusive.
    '''
    features_model = load_model(modelPath, compile=False)
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper

    type_exp = 'N-MEV'

    # KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    params = {'k': n_shot,
              'weights': 'distance',
              'p': 2,
              'metric': 'cosine'
              }

    print("N_shot: ", n_shot)
    acc_list_Top1, acc_list_Top5 = [], []
    exp_times = 2
    total_time = 0

    for i in range(exp_times):
        # signature_dict, test_dict, sites = utility.getRFdataDict(opts.input, n_shot, data_dim, train_pool_size=20, test_size=70)
        signature_dict, test_dict, sites = getRFdataDict(opts.testData, opts, n_shot, n_instance=(2000 + max_n), max_n=max_n)
        if i == 0:
            size_of_problem = len(list(test_dict.keys()))
            print("Size of Problem: ", size_of_problem)
        signature_vector_dict, test_vector_dict = create_test_set_Wang_disjoint(signature_dict,
                                                                                test_dict,
                                                                                sites,
                                                                                features_model=features_model,
                                                                                type_exp=type_exp)
        # Measure the performance (accuracy)
        start = time.time()
        acc_knn_top1, acc_knn_top5 = kNN_accuracy(signature_vector_dict, test_vector_dict, params)
        end = time.time()
        total_time += (end-start)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))

    print(str(acc_list_Top1).strip('[]'))
    print(str(acc_list_Top5).strip('[]'))
    rtnLine = 'n_shot: {}\tacc for top 1: {} and std is: {}\n'.format(n_shot, mean(acc_list_Top1), stdev(acc_list_Top1))
    rtnLine = rtnLine + '\nacc for top 5: {} and std is: {}\n'.format(mean(acc_list_Top5), stdev(acc_list_Top5))
    rtnLine = rtnLine + '\nKNN training time : {}\n'.format(total_time/exp_times)
    print(rtnLine)
    return rtnLine


def run(opts):
    source = "day1"
    outfile = os.path.join(ResDir, 'ccs19_target{}_results.txt'.format(source))
    f = open(outfile, 'a+')
    print('\n\n#################### test time is: {} #########################'.format(time.ctime()), file=f)
    tsnList = [100, 200, 400, 800, 1600]
    modelPath = "/home/haipeng/Documents/rf_lhp/TF/res_out/modelDir/baseline_triplet_day1_500_True"
    for tsn in tsnList:
        rtnLine = Wang_Disjoint_Experment(opts, modelPath, n_shot=tsn, max_n=max(tsnList))
        print(rtnLine, file=f)
    f.close()


class tfOpts():
    def __init__(self, source_path, test_path, location, file_key='*.bin', num_slice = 3600, start_ix=0, slice_len=288, stride=1):
        self.root_dir = source_path
        self.testData = test_path
        self.semiHard = True
        self.plotModel = False
        self.useGpu = True
        self.testType = "tsn"
        # self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.channel_first = False
        self.sample = True


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelPath', help='')
    parser.add_argument('-t', '--exp_type', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = tfOpts(source_path="/home/haipeng/Documents/dataset/radio_dataset/test_dataset",
                  test_path="/home/haipeng/Documents/dataset/radio_dataset/test_dataset_2",
                  location='symbols')
    run(opts)
