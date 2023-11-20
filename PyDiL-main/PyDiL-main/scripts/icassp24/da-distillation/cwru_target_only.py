import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

from tqdm.auto import tqdm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

fname = 'cwru_target_only.csv'

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/da-distillation')
data_path = os.path.join(
    base_path,
    'data',
    'crwu',
    'features',
)


domain_names = ['A', 'B', 'C']

n_dim = 256
n_classes = 10
n_domains = 3

for target_domain in domain_names:
    sources = [d for d in domain_names if d != target_domain]
    with open(
        os.path.join(data_path,
                     f'mlp_fts_256_target_{target_domain}.pkl'), 'rb') as f:
        dataset = pickle.loads(f.read())

    Xt_tr = dataset[target_domain]['fold 0']['Train']["Features"]
    Yt_tr = dataset[target_domain]['fold 0']['Train']["Labels"].float()
    Xt_ts = dataset[target_domain]['fold 0']['Test']["Features"]
    Yt_ts = dataset[target_domain]['fold 0']['Test']["Labels"].float()
    # Xt = (Xt - Xt.mean()) / Xt.std()
    print('... Target {}'.format(target_domain))
    print('...... Train: {}'.format(Xt_tr.shape))
    print('...... Test:  {}'.format(Xt_ts.shape))

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
