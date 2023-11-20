import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

from pydil.utils.parsing import parse_args_distribution_matching  # noqa: E402

from tqdm.auto import tqdm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_distribution_matching()

fname = 'tep_target_only.csv'

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/da-distillation')
tep_path = os.path.join(
    base_path,
    'data',
    'tennessee-eastman',
    'features',
    'feature_engineering',
    'TEPFeatures.pickle'
)

# Reads the data
with open(tep_path, 'rb') as f:
    dataset = pickle.loads(f.read())

domain_names = [f'Mode {k}' for k in range(1, 7)]

X, y, d = [], [], []
for mode in range(1, 7):
    Xm = dataset[f'Mode {mode}']['Features']
    Ym = dataset[f'Mode {mode}']['Labels']
    ym = Ym.argmax(axis=1)

    X.append(torch.from_numpy(Xm).float())
    y.append(torch.from_numpy(ym).float())
    d.append(torch.Tensor([mode] * len(Xm)))
X = torch.cat(X, dim=0)
y = torch.cat(y, dim=0)
Y = torch.nn.functional.one_hot(y.long(), num_classes=29)
Y = Y.float()
d = torch.cat(d, dim=0)
d = d - d.min()

n_dim = X.shape[1]
n_classes = 29
n_domains = 6
Copt = 10 ** (14 / 3)

for target_domain in range(n_domains):
    ind = torch.where(d == target_domain)[0]
    ind_tr, ind_ts = train_test_split(ind,
                                      train_size=0.8,
                                      random_state=7,
                                      stratify=y[ind])
    Xt_tr, yt_tr, Yt_tr = X[ind_tr], y[ind_tr], Y[ind_tr]
    Xt_ts, yt_ts, Yt_ts = X[ind_ts], y[ind_ts], Y[ind_ts]

    μt, σt = Xt_tr.mean(), Xt_tr.std()
    Xt_tr = (Xt_tr - μt) / σt
    Xt_ts = (Xt_ts - μt) / σt

    accs = []
    accs_err = []

    ipc_range = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    mperf, sperf = [], []
    for ipc in ipc_range:
        acc = []
        for _ in tqdm(range(5)):
            xsyn = []
            ysyn = []
            for c in range(n_classes):
                ind = np.where(Yt_tr.argmax(dim=1) == c)[0]
                ind = np.random.choice(ind, size=ipc)
                xsyn.append(Xt_tr[ind])
                ysyn.append(Yt_tr[ind])
            xsyn = torch.cat(xsyn, dim=0)
            ysyn = torch.cat(ysyn, dim=0)
            clf = SVC(kernel='linear', max_iter=int(1e+6), C=1)
            clf.fit(xsyn, ysyn.argmax(dim=1))

            y_pred = clf.predict(Xt_ts)
            acc.append(100 * accuracy_score(y_pred, Yt_ts.argmax(dim=1)))
        m = np.mean(acc)
        s = np.std(acc)
        print(f"IPC {ipc}, perf: {m} ± {s}")

        with open(os.path.join(results_path, fname), 'a') as f:
            f.write((f'{domain_names[target_domain]},{ipc},'
                     f'{m},{s}\n'))
