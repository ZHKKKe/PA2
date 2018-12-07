import os
import math
import random
import logging
import scipy.io
import numpy as np


def mat_data_loader(dpath, problem_id):
    if not dpath.endswith('.mat'):
        logging.error('wrong format: {0} for mat data loader'.format(dpath.split('.')[-1]))
        return None
    elif not os.path.isfile(dpath):
        logging.error('cannot find dataset path: {0}'.format(dpath))
        return None

    data = {}
    raw_data = scipy.io.loadmat(dpath)
    if problem_id == 1:
        data['dataA_X'] = np.asarray(raw_data['dataA_X'])
        data['dataA_Y'] = np.asarray(raw_data['dataA_Y'])
        data['dataB_X'] = np.asarray(raw_data['dataB_X'])
        data['dataB_Y'] = np.asarray(raw_data['dataB_Y'])
        data['dataC_X'] = np.asarray(raw_data['dataC_X'])
        data['dataC_Y'] = np.asarray(raw_data['dataC_Y'])

    return data


if __name__ == '__main__':
    mat_data_loader('../data/PA2-cluster-data/cluster_data.mat', 1)