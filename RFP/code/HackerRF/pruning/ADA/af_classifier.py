#!/usr/bin/env python3.6

import os
import sys
import argparse
import time
from statistics import mean, stdev

import numpy as np
import utility
from sklearn.metrics import accuracy_score
import load_slice_IQ
import af_model
import af_optimizer
import af_data
import af_test as test
from tensorflow.keras.utils import to_categorical

RootDir = os.getcwd()
toolsDir = os.path.join(RootDir, 'tools')
# import utility
import tools as mytools

thisFile = os.path.abspath(__file__)
currentDir = os.path.dirname(thisFile)
ResDir = os.path.join(currentDir, 'ADA/res_out')
modelDir = os.path.join(ResDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


def train(param, args):
    source = os.path.basename(args.source).split('.')[0]
    target = os.path.basename(args.target).split('.')[0]
    # setup model
    inp_shape = param["inp_dims"]
    embsz = param['embsz']
    # inp, embedding = af_model.build_embedding_DF_model(inp_shape, embsz)
    print("modelType: {}".format(args.modelType))
    if args.modelType == "DF":
        inp, embedding = af_model.build_embedding_DF_model(inp_shape, embsz)
    elif args.modelType == "homegrown":
        inp, embedding = af_model.build_embedding_Homegrown(inp_shape, embsz)
    classifier = af_model.build_classifier_conv(param, embedding)
    discriminator = af_model.build_discriminator_conv(param, embedding)

    combined_classifier = af_model.build_combined_classifier(inp, classifier)
    combined_discriminator = af_model.build_combined_discriminator(inp, discriminator)
    combined_model = af_model.build_combined_model(inp, [classifier, discriminator])

    combined_classifier.compile(optimizer=af_optimizer.opt_classifier(param),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
    combined_discriminator.compile(optimizer=af_optimizer.opt_discriminator(param),
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])

    loss_dict = {}
    loss_dict['class_act_last'] = 'categorical_crossentropy'
    loss_dict['dis_act_last'] = 'binary_crossentropy'

    loss_weight_dict = {}
    loss_weight_dict['class_act_last'] = param["class_loss_weight"],
    loss_weight_dict['dis_act_last'] = param["dis_loss_weight"]

    combined_model.compile(optimizer=af_optimizer.opt_combined(param),
                           loss=loss_dict,
                           loss_weights=loss_weight_dict,
                           metrics=['accuracy'])

    if args.plotModel:
        from keras.utils import plot_model
        plot_model(combined_model, to_file='multi_model_{}.png'.format(inp_shape[0]), dpi=200)
        sys.exit(1)

    # load the data
    Xs1, ys1 = param["source_data"], param["source_label"]
    Xt, yt = param["target_data"], param["target_label"]

    param['batch_size'] = len(Xt) if len(Xt) < param['batch_size'] else param['batch_size']

    # Source domain is represented by label 0 and Target by 1

    ys_adv1 = np.array(([0.] * param["batch_size"]))
    yt_adv = np.array(([1.] *  param["batch_size"]))

    y_advb_1 = np.array(([0] * param["batch_size"] + [1] *  param["batch_size"]))  # For gradient reversal
    y_advb_2 = np.array(([1] * param["batch_size"]+ [0] *  param["batch_size"]))

    weight_class = np.array(([1] * param["batch_size"] + [0] * param["batch_size"]))
    weight_adv = np.ones((param["batch_size"] * 2,))

    S_1_batches = af_data.batch_generator([Xs1, ys1], param["batch_size"])
    T_batches = af_data.batch_generator([Xt, np.zeros(shape=(len(Xt),))],  param["batch_size"])

    # start the training
    start = time.time()
    logs = []
    for i in range(param["num_iterations"]):
        Xsb1, ysb1 = next(S_1_batches)
        Xtb, ytb = next(T_batches)
        X_adv = np.concatenate([Xsb1, Xtb])
        y_class1 = np.concatenate([ysb1, np.zeros_like(ysb1)])

        # 'Epoch {}: train the classifier'.format(i)
        adv_weights = []
        for layer in combined_model.layers:
            if (layer.name.startswith("dis_")):
                adv_weights.append(layer.get_weights())
        stats1 = combined_model.train_on_batch(X_adv, [y_class1, y_advb_1], sample_weight=[weight_class, weight_adv])

        k = 0
        for layer in combined_model.layers:
            if (layer.name.startswith("dis_")):
                layer.set_weights(adv_weights[k])
                k += 1

        # 'Epoch {}: train the discriminator'.format(i)
        class_weights = []
        for layer in combined_model.layers:
            if (not layer.name.startswith("dis_")):
                class_weights.append(layer.get_weights())
        stats2 = combined_discriminator.train_on_batch(X_adv, y_advb_2)

        k = 0
        for layer in combined_model.layers:
            if (not layer.name.startswith("dis_")):
                layer.set_weights(class_weights[k])
                k += 1

        # show the intermediate results
        if ((i + 1) % param["test_interval"] == 0):

            ys1_pred = combined_classifier.predict(Xsb1)
            # yt_pred = combined_classifier.predict(Xt)
            ys1_adv_pred = combined_discriminator.predict(Xsb1)
            yt_adv_pred = combined_discriminator.predict(Xtb)

            source1_accuracy = accuracy_score(ysb1.argmax(1), ys1_pred.argmax(1))
            source_domain1_accuracy = accuracy_score(ys_adv1, np.argmax(ys1_adv_pred, axis=1))
            target_domain_accuracy = accuracy_score(yt_adv, np.argmax(yt_adv_pred, axis=1))

            log_str = ["iter: {:05d}:".format(i),
                       "LABEL CLASSIFICATION: source_1_acc: {:.5f}".format(source1_accuracy * 100),
                       "DOMAIN DISCRIMINATION: source_domain1_accuracy: {:.5f}, target_domain_accuracy: {:.5f} \n".format(source_domain1_accuracy * 100, target_domain_accuracy * 100)]
            log_str = '\n'.join(log_str)
            print(log_str + '\n')
            logs.append(log_str)

    last = time.time() - start
    tmpLine = 'total training time is: {:f} sec\n'.format(last)
    logs.append(tmpLine)
    contents = '\n'.join(logs)
    reportPath = os.path.join(ResDir, 'trainReport_oneClassifer_source_{}_target_{}.txt'.format(source, target))
    with open(reportPath, 'w') as f:
        f.write(contents)
    classifier_path = os.path.join(modelDir, "ADA_{}_{}_{}_sourceSize_{}_targetSize_{}".format(args.modelType, source, args.dataType, len(Xs1)//10, len(Xt)//10))
    combined_classifier.save(classifier_path)

    return classifier_path, last


def run(param, args):
    source = os.path.basename(args.source).split('.')[0]
    target = os.path.basename(args.target).split('.')[0]
    flag = False if 'trainNum' == args.testType else True
    test_num = 2
    #NUM_CLASS = 10
    if flag:
        # Load source and target data
        # param["source_data"], param["source_label"] = data.data_loader(args.source, param["inp_dims"], sample_num=25)
        dataOpts = load_slice_IQ.loadDataOpts(args.source, args.location, num_slice=args.num_slice, slice_len=args.slice_len,
                                          start_idx = args.start_ix, stride = args.stride, mul_trans = True, window = args.window, dataType=args.dataType)
        param["source_data"], param["source_label"], _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        # Encode labels into one-hot format
        # clsNum = len(set(param["source_label"]))
        param["source_label"] = to_categorical(param["source_label"], NUM_CLASS)
    else:
        print('will run train num test, so not loading training data at first')

    if 'nShot' == args.testType:
        print('run n_shot test...')
        n_shot_list = [100,200,400,800]
        n_shot_list = n_shot_list[::-1]
        #n_shot_list = [20]
        outfile = os.path.join(ResDir, 'ADA_one_source_{}_target_{}_res.txt'.format(source, target))
        f = open(outfile, 'a+')
        print('\n\n##################### dataType:{}, test time is: {}####################'.format(args.dataType,time.ctime()), file=f, flush=True)
        for n_shot in n_shot_list:
            acc_list = []
            signature_dict, test_dict, _ = utility.getRFdataDict(args.target, args, n_shot=n_shot, n_instance=1000+max(n_shot_list),
                                                                 max_n=max(n_shot_list))
            target_data, target_label = mytools.datadict2data(signature_dict)
            print('target data shape: ', target_data.shape)
            # target_data = target_data[:, :, np.newaxis]
            target_label = af_data.one_hot_encoding(target_label, len(set(target_label)))
            param["target_data"], param["target_label"] = target_data, target_label
            model_path, time_last = train(param, args)
            time_list = []
            # time_last_list.append(time_last)
            for i in range(test_num):
                # Train phase
                #signature_dict, test_dict, _ = utility.getRFdataDict(args.target, args, n_shot = n_shot, n_instance=2000 + max(n_shot_list), max_n=max(n_shot_list))

                # Test phase
                test_opts = test.MyOpts(model_path, nShot=n_shot, tuning=True, aug=0, exp_type=args.exp_type)
                test_opts.nShot = n_shot
                test_params = test.generate_default_params(test_opts)
                test_params['n_shot'] = n_shot
                inp_shape = param["inp_dims"]
                acc, fine_tune_time = test.run(test_opts, args.modelType, signature_dict, test_dict, params=test_params, emb_size=param['embsz'], inp_shape=inp_shape)
                acc_list.append(acc)
                time_list.append(fine_tune_time)
                print('acc of source {} and target {} with n_shot {} on {} test dataset is: {:f}, test_time: {}'.format(source, target, n_shot, len(test_dict[0]), acc, fine_tune_time))
            print('acc of source {} and target {} with n_shot {} is: {:f}, stdev is: {:f}, time last: {:f}\n\n'.format(source, target, n_shot, mean(acc_list), stdev(acc_list), time_last))
            resLine = 'modelType: {}, acc of source {} and target {} with n_shot {} is: {:f}, stdev is: {:f}, pre_training time last: {:f}, training time: {}\n\n'.format(args.modelType, source, target, n_shot, mean(acc_list), stdev(acc_list), time_last, mean(time_list))
            print(resLine, file=f, flush=True)
        f.close()
    elif 'aug' == args.testType:
        print('will run aug test...')
        pass
    elif 'trainNum' == args.testType:
        print('will run train num test...')
        n_shot = 20
        outfile = os.path.join(ResDir, 'trainNumTest_ADA_one_source_{}_target_{}_res.txt'.format(source, target))
        f = open(outfile, 'a+')
        print('\n\n################### test time is: {} ####################'.format(time.ctime()), file=f, flush=True)
        print('test with N shot num: {}'.format(n_shot), file=f, flush=True)
        trainNumList = [25, 50, 75, 100, 125]
        for trainNum in trainNumList:
            acc_list, time_last_list = [], []
            # load training data accord to the train num
            param["source_data"], param["source_label"] = af_data.data_loader(args.source, param["inp_dims"], sample_num=trainNum)
            print('train data shape is: ', np.array(param['source_data']).shape)
            clsNum = len(set(param["source_label"]))
            param["source_label"] = af_data.one_hot_encoding(param["source_label"], clsNum)

            for i in range(test_num):
                # Train phase
                signature_dict, test_dict, sites = utility.getDataDict(args.target, n_shot=n_shot, data_dim=param['inp_dims'], train_pool_size=20, test_size=70)
                target_data, target_label = mytools.datadict2data(signature_dict)
                target_data = target_data[:, :, np.newaxis]
                target_label = af_data.one_hot_encoding(target_label, len(set(target_label)))
                param["target_data"], param["target_label"] = target_data, target_label
                model_path, time_last = train(param, args)
                time_last_list.append(time_last)

                # Test phase
                test_opts = test.MyOpts(model_path, nShot=n_shot, tuning=True, aug=0, exp_type=args.exp_type)
                test_opts.nShot = n_shot
                test_params = test.generate_default_params(test_opts)
                inp_shape = (param["inp_dims"], 1)
                _, acc = test.run(test_opts, signature_dict, test_dict, params=test_params, emb_size=param['embsz'], inp_shape=inp_shape, test_times=1)
                acc_list.append(acc)
                print('acc of source {} and target {} with n_shot {} on {} test dataset is: {:f}'.format(source, target, n_shot, len(test_dict), acc))
            resLine = 'acc of source {} and target {} with n_shot {} is: {:f}, stdev is: {:f}, training time last: {:f}'.format(source, target, n_shot, mean(acc_list), stdev(acc_list), mean(time_last_list))
            print(resLine, file=f, flush=True)
        f.close()
    elif 'trainTime' == args.testType:
        # Train phase
        n_shot = 20
        signature_dict, test_dict, sites = utility.getDataDict(args.target, n_shot=n_shot, data_dim=param['inp_dims'], train_pool_size=20, test_size=70)
        target_data, target_label = mytools.datadict2data(signature_dict)
        target_data = target_data[:, :, np.newaxis]
        target_label = af_data.one_hot_encoding(target_label, len(set(target_label)))
        param["target_data"], param["target_label"] = target_data, target_label
        model_path, time_last = train(param, args)
        print('training time last: ', time_last)

    else:
        raise


def generate_params():
    # Initialize parameters
    param = {}
    param["number_of_gpus"] = 1
    param["network_name"] = 'self_define'
    param["inp_dims"] = (288,2)
    param["num_iterations"] = 10000  # training epoch numbers, default as 1000

    #'--lr_classifier' = "Learning rate for classifier model"
    #'--b1_classifier' = "Exponential decay rate of first moment for classifier model optimizer"
    #'--b2_classifier' = "Exponential decay rate of second moment for classifier model optimizer"
    param["lr_classifier"] = 0.0001
    param["b1_classifier"] = 0.9
    param["b2_classifier"] = 0.999

    #'--lr_discriminator' = "Learning rate for discriminator model")
    #'--b1_discriminator' = "Exponential decay rate of first moment for discriminator model optimizer"
    #'--b2_discriminator' = "Exponential decay rate of second moment for discriminator model optimizer"
    param["lr_discriminator"] = 0.00001
    param["b1_discriminator"] = 0.9
    param["b2_discriminator"] = 0.999

    #'--lr_combined'  "Learning rate for combined model"
    #'--b1_combined'  "Exponential decay rate of first moment for combined model optimizer"
    #'--b2_combined'  "Exponential decay rate of second moment for combined model optimizer"
    param["lr_combined"] = 0.00001
    param["b1_combined"] = 0.9
    param["b2_combined"] = 0.999

    param["batch_size"] = 128
    param["test_interval"] = 1000

    # params for search
    param['cls_depth'] = 1
    param['dis_depth'] = 1
    param["class_loss_weight"] = 4
    param["dis_loss_weight"] = 4
    param["drop_classifier"] = 0.4
    param["drop_discriminator"] = 0.4
    param['embsz'] = 64

    param['dis_act'] = 'softsign'
    param['cls_act'] = 'softsign'

    param['dis_conv_1'] = 64
    param['dis_kernel_1'] = 4
    param['dis_pool_1'] = 4

    param['dis_conv_2'] = 128
    param['dis_kernel_2'] = 4
    param['dis_pool_2'] = 4

    param['cls_conv_1'] = 64
    param['cls_kernel_1'] = 4
    param['cls_pool_1'] = 4

    param['cls_conv_2'] = 128
    param['cls_kernel_2'] = 4
    param['cls_pool_2'] = 4

    param['dis_dense2'] = 128
    param['cls_dense2'] = 128
    return param


class MyOpts():
    def __init__(self, source, target, plotModel):
        self.source = source
        self.target = target
        self.plotModel = plotModel


def parseArgs(argv):
    # Read parameter values from the console
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('-s', '--source', help='can input single or multiple source')
    parser.add_argument('-t', '--target', help="Path to target dataset")
    parser.add_argument('-p', '--plotModel', action='store_true', help="options to plot the model shape")
    parser.add_argument('-g', '--useGpu', action='store_true', help='use gpu or not')
    parser.add_argument('-e', '--exp_type', action='store_true', help='')
    parser.add_argument('-tt', '--testType', default='nShot', help='choose which test to run: nShot/aug/trainNum')

    args = parser.parse_args()
    return args


class afOpts():
    def __init__(self, source_path, test_path, location, file_key='*.bin', num_slice=10000, start_ix=0, slice_len=288, stride=1, modelType='DF', window=64, dataType = 'IQ'):
        self.source = source_path
        self.target = test_path
        self.plotModel = False
        self.useGpu = True
        self.testType = "nShot"
        # self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.exp_type = False
        self.mul_trans = True 
        self.modelType = modelType
        self.window = window
        self.dataType = dataType

if __name__ == "__main__":
    # args = parseArgs(sys.argv)
    dataPath = "/home/haipeng/Documents/dataset/radio_dataset/"
    source = ['neu_different_day3']
    target = ['neu_different_day4'] 
    dataSet = zip(source, target)
    for p in dataSet:
        for s in [288]:
            args = afOpts(source_path=dataPath + p[0], test_path=dataPath + p[1], location="after_equ", stride = 288, modelType = "DF", window=64, dataType = 'IQ')
            # Set GPU device
            if args.useGpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'

            param = generate_params()
            run(param, args)
