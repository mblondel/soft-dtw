import os.path
import numpy as np


home = os.path.expanduser("~")
data_dir = os.path.join(home, "sdtw_data")
ucr_dir = os.path.join(data_dir, "UCR_TS_Archive_2015")


def _parse_ucr(filename):
    y = []
    X = []
    for line in open(filename):
        line = line.strip()
        arr = line.split(",")
        label = int(arr[0])
        feat = list(map(float, arr[1:]))
        feat = np.array(feat).reshape(-1, 1)
        y.append(label)
        X.append(feat)
    return X, np.array(y)


def list_ucr():
    return sorted(os.listdir(ucr_dir))


def load_ucr(name):
    folder = os.path.join(ucr_dir, name)
    tr = os.path.join(folder, "%s_TRAIN" % name)
    te = os.path.join(folder, "%s_TEST" % name)

    try:
        X_tr, y_tr = _parse_ucr(tr)
        X_te, y_te = _parse_ucr(te)
    except IOError:
        raise IOError("Please copy UCR_TS_Archive_2015/ to $HOME/sdtw_data/. "
                      "Download from www.cs.ucr.edu/~eamonn/time_series_data.")

    y_tr = np.array(y_tr)
    y_te = np.array(y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)

    return X_tr, y_tr, X_te, y_te
