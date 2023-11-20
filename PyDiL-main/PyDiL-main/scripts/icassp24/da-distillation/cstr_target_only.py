import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

from scipy.fft import fft  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

fname = 'cstr_target_only.csv'

base_path = '/home/efernand/data/CSTR'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/da-distillation')
n_dim = 7
n_classes = 13
dataset = np.load(os.path.join(base_path, 'cstr_rawdata.npy'))
X = dataset[:, :-4]
y = dataset[:, -4]
Y = torch.nn.functional.one_hot(
    torch.from_numpy(y).long(),
    num_classes=n_classes
)
d = dataset[:, -3]
parameter_noise = dataset[:, -2]
reaction_order = dataset[:, -1]

domain_names = []
unique_domains = np.unique(d)
for domain in np.unique(d):
    ind = np.where(d == domain)[0]
    N = np.unique(reaction_order[ind]).item()
    ϵ = np.unique(parameter_noise[ind]).item()

    domain_names.append(f'N_{N}_ϵ_{ϵ}')

X = np.stack([X[:, i * 200: (i + 1) * 200] for i in range(7)])
Z = np.stack([(abs(fft(X[v, ...])) ** 2).sum(axis=1) for v in range(7)]).T
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)

X = torch.from_numpy(Z).float()
y = torch.from_numpy(y).long()
Y = torch.nn.functional.one_hot(y, num_classes=13)
d = torch.from_numpy(d).float()

n_dim = 7
n_domains = 7
n_classes = 13

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
            f.write((f'{target_domain},{ipc},'
                     f'{m},{s}\n'))
