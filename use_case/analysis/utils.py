import pickle
import numpy as np


def extract_initial_data(log_pickle, nmb_samples=None,
                         low_fidelity='10it', high_fidelity='40it'):
    """

    :param log_pickle: path to the pickle file with logs
    :param nmb_samples: None (if take all) or int (if number of first sample in the file)
    :param low_fidelity: name of the low fidelity, e.g., '10it' or '1pm'
    :param high_fidelity: name of the high fidelity, e.g., '40it' or '1pct'
    :return:
    """

    x_low = []
    y_low = []
    x_high = []
    y_high = []

    dump = pickle.load(open(log_pickle, "rb"))
    if nmb_samples is None:
        logs = dump['log']
    else:
        logs = dump['log']
    for i in range(len(logs)):
        fid = logs[i]['annotations']['fidelity_level']
        if fid == low_fidelity:
            x_low.append(logs[i]['parameters'])
            y_low.append(logs[i]['objective'])
        if fid == high_fidelity:
            x_high.append(logs[i]['parameters'])
            y_high.append(logs[i]['objective'])

    x_low = x_low[:nmb_samples]
    x_high = x_high[:nmb_samples]
    y_low = y_low[:nmb_samples]
    y_high = y_high[:nmb_samples]

    x_initial = np.vstack((np.c_[np.array(x_low), np.zeros(len(x_low))],
                           np.c_[np.array(x_high), np.ones(len(x_high))]))
    y_initial = np.array(y_low + y_high).reshape(-1,1)

    return x_initial, y_initial