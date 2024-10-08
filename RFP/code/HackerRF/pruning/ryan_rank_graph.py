
import sys
import numpy as np
import pandas as pd
from statistics import mean, stdev
import load_slice_IQ
sys.path.append('models')

import tools as mytools
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def class_ranks(model, testX, testY, num_class, preds=None):
    ranks = list()
    testY = np.array(testY)
    for i in range(num_class):
        ids = np.where(testY == i)
        if preds is None:
            OneClsData = np.squeeze(np.take(testX, ids, axis=0))
            OneClsPreds = model.predict(OneClsData, verbose=1)
        else:
            OneClsPreds = np.take(preds, ids, axis=0)
        OneCls_rank = accumulate_ranks(np.squeeze(OneClsPreds), num_class, i)

        ranks.append(OneCls_rank)
    return ranks





