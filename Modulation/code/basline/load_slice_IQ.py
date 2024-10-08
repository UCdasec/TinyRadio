import os
import sys
import glob
import argparse
import pdb
import random
import numpy as np
from scipy import signal
from tensorflow.keras.utils import to_categorical
import utility
import math
import matplotlib.pyplot as plt

# m = np.complex(2,2)

# n = np.abs(m)
# print()

def normalizeData(v):
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    v = (v - v_min)/(v_max - v_min)
    return v


def STFT(iq_data, seg_len):
    # f, t, Zxx = signal.spectrogram(iq_data)

    f, t, Zxx = signal.stft(iq_data, nperseg=seg_len, fs=2000000, return_onesided=False)
    # Zxx = Zxx.transpose()
    Z_abs = np.abs(Zxx)
    #power = 10*np.log(Z_abs)
    power = 1000*np.power(Z_abs, 2)
    # amp = -30
    # plt.pcolormesh(t[:10000], f, power[:,:10000], vmin=-300, vmax=amp, shading='gouraud')
    # plt.title('seg_len=64, fs=2M')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # plt.savefig('real_imag_seq.pdf',format="pdf")
    return power


def read_spec_bin(filename, seg_len):
    with open(filename, 'rb') as bin_f:
        iq_seq = np.fromfile(bin_f, dtype=np.complex64)
        IQ_data = STFT(iq_seq, seg_len)
        #print(len(iq_seq), IQ_data.shape)
    return IQ_data


def read_f32_bin(filename, start_idx, channel_first):
    with open(filename, 'rb') as bin_f:
        iq_seq = np.fromfile(bin_f, dtype='<f4')
        n_samples = iq_seq.shape[0] // 2
        IQ_data = np.zeros((2, n_samples))
        IQ_data[0, :] = iq_seq[range(0, iq_seq.shape[0]-1, 2)]
        IQ_data[1, :] = iq_seq[range(1, iq_seq.shape[0], 2)]

    del iq_seq
    rtn_data = IQ_data[:, start_idx:]
    print(rtn_data.shape)
    if not channel_first:
        rtn_data = rtn_data.T

    return rtn_data


def dev_bin_dataset(glob_dat_path, n_slices_per_dev, slice_len, start_idx, channel_first, sample, mul_trans, dataType, stride, window):
    filelist = sorted(glob.glob(glob_dat_path))
    #random.shuffle(filelist)
#    filelist = os.listdir(glob_dat_path)
    num_tran = len(filelist)
    axis = 1 if channel_first else 0
    all_IQ_data = []

    print("mul_trans or not: {}, sample data or not: {}, start_idx: {}".format(mul_trans, sample, start_idx))
    samples_per_tran_list = []
    if not mul_trans:
        samples_per_tran_list = [n_slices_per_dev]
    else:
        samples_per_tran_list = [n_slices_per_dev // num_tran + 1]*num_tran
        #k = np.random.dirichlet(np.ones(num_tran), size = 1)[0]
        #samples_per_tran_list = [math.ceil(s*n_slices_per_dev) for s in k]
        #print(sum(samples_per_tran_list))

    #print("num of transmissions: {}, traces per transmission: {}".format(len(filelist), samples_per_tran))
    for i,f in enumerate(filelist):
        samples_per_tran = samples_per_tran_list[i]
        if dataType == "IQ":
            IQs_per_tran = read_f32_bin(f, start_idx, channel_first)
            slices_per_tran = []
            if stride == 'r':
                pool_size = max(IQs_per_tran.shape[0], IQs_per_tran.shape[1]) - slice_len - 1
                IQ_per_tran_idx = sorted(random.sample(list(range(pool_size)), samples_per_tran))
            else:
                IQ_per_tran_idx = [i*stride for i in range(samples_per_tran)]
            
            print("file: {}, samples_num:{}, {}".format(f, samples_per_tran,IQ_per_tran_idx[:3]))

            if channel_first:
                for i in IQ_per_tran_idx:
                    slices_per_tran.append(IQs_per_tran[:, i:i+slice_len])
            else:
                for i in IQ_per_tran_idx:
                    slices_per_tran.append(IQs_per_tran[i:i+slice_len, :])

           # print(len(slices_per_tran))
            if len(all_IQ_data):
                if len(np.array(slices_per_tran).shape) == 1 or len(np.array(all_IQ_data).shape) == 1:
                    print()
                all_IQ_data = np.concatenate((all_IQ_data, slices_per_tran), axis=axis)
            else:
                all_IQ_data = slices_per_tran

        elif dataType == "spectrogram":
            print("file: {}, samples_num:{}".format(f, samples_per_tran))
            spec_per_tran = read_spec_bin(f, window)[start_idx:]
            spec_len = spec_per_tran.shape[1]

            spec_chunks = [spec_per_tran[:, i:i+slice_len] for i in range(0, spec_len-slice_len, slice_len)]
            print(len(spec_chunks))
            if len(all_IQ_data):
                all_IQ_data = np.concatenate((all_IQ_data, spec_chunks[:samples_per_tran]), axis=0)
            else:
                all_IQ_data = spec_chunks[:samples_per_tran]

            # all_IQ_data.append(spec_per_tran)
        if not mul_trans:
            all_IQ_data = np.stack(all_IQ_data, axis=0)
            break
    return all_IQ_data, num_tran


def loadData(args, split=True):
    print('loading data from {}'.format(args.root_dir))
    n_slices_per_dev = args.num_slice
    start_idx = args.start_idx
    file_key = args.file_key
    slice_len = args.slice_len
    channel_first = args.channel_first
    window = args.window
    dataType = args.dataType
    dev_dir_list = []
    dev_dir_names = sorted(os.listdir(args.root_dir))
    print(dev_dir_names)
    for n in dev_dir_names:
        tmp = os.path.join(args.root_dir, n)
        dev_dir_list.append(tmp)

    stride = args.stride
    n_devices = len(dev_dir_list)
    print("number of devices: {}".format(n_devices))
    # locations = ["after_fft","before_fft", "output_equ", "symbols"]
    # locations = ["output_equ"]

    # if channel_first:
    #     slice_dims = (2, args.slice_len)
    #     samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[1]
    # else:
    #     slice_dims = (args.slice_len, 2)
    #     samps_to_retrieve = (n_slices_per_dev - 1) * stride + slice_dims[0]

    x_train, y_train, x_test, y_test = [], [], [], []
    if split:
        split_ratio = {'train': 0.8, 'val': 0.2}
    else:
        split_ratio = {'train': 1.0, 'val': 0.0}
    for i, d in enumerate(dev_dir_list):

        p = os.path.join(d, args.location)
        X_data_pd, num_tran = dev_bin_dataset(os.path.join(p, file_key), n_slices_per_dev, slice_len, start_idx, channel_first, args.sample, args.mul_trans, dataType, stride, window)
       # X_data_pd, num_tran = dev_bin_dataset(p, n_slices_per_dev, slice_len, start_ix, channel_first, args.sample, args.mul_trans)
        #y_data_pd = i * np.ones(n_slices_per_dev, )
        y_data_pd = i * np.ones(X_data_pd.shape[0], )
        print(X_data_pd.shape)
        x_train_pd, y_train_pd, x_test_pd, y_test_pd = utility.splitData(split_ratio, X_data_pd, y_data_pd)

        if i == 0:
            x_train, x_test = x_train_pd, x_test_pd
            y_train, y_test = y_train_pd, y_test_pd
        else:
            x_train = np.concatenate((x_train, x_train_pd), axis=0)
            x_test = np.concatenate((x_test, x_test_pd), axis=0)
            y_train = np.concatenate((y_train, y_train_pd), axis=0)
            y_test = np.concatenate((y_test, y_test_pd), axis=0)
        del X_data_pd


    return x_train, y_train, x_test, y_test, n_devices


class loadDataOpts():
    def __init__(self, root_dir, location, file_key='*.bin', num_slice=10000, start_idx=0, slice_len=144, stride=144, channel_first = False, sample = False, mul_trans = True, window = 64, dataType = "spectrum"):
        self.root_dir = root_dir
        self.num_slice = num_slice
        self.start_idx = start_idx
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.channel_first = channel_first
        self.sample = sample
        self.mul_trans = mul_trans
        self.dataType = dataType
        self.window = window


def parseArgs(argv):
    Desc = 'Read and slice the collected I/Q samples'
    parser = argparse.ArgumentParser(description=Desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--root_dir', required=True, help='Root directory for the devices\' folders.')
    parser.add_argument('-n', '--num_slice', required=True, type=int, help='Number of slices to be generated for each device.')
    parser.add_argument('-i', '--start_ix', type=int, default=0, help='Starting read index in .bin files.')
    parser.add_argument('-l', '--slice_len', type=int, default=288, help='Lenght of slices.')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride used for windowing.')
    parser.add_argument('-f', '--file_key', default='*.bin', help='used to choose different filetype, choose from *.bin/*.sigmf-meta')
    parser.add_argument('-cf', '--channel_first', action='store_true', help='if set channel first otherwise channel last')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = loadDataOpts(root_dir="/home/haipeng/Documents/dataset/radio_dataset/our_day1", location='before_fft', dataType="spectrum")
    # channel_first = False
    x_train, y_train, x_test, y_test, NUM_CLASS = loadData(opts,split=True)
    # np.savez("neu_day1_afterEqu_288features_100k", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print('train data shape: ', x_train.shape, 'train label shape: ', y_train.shape)
    print('test data shape: ', x_test.shape, 'test label shape: ', y_test.shape)
    print('all test done!')
