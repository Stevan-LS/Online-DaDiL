import os
import sys

sys.path.append('./')

import json  # noqa: E402
import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

from pydil.utils.parsing import parse_args_distribution_matching  # noqa: E402
from pydil.ipms.mmd_ipms import ClassConditionalMMD  # noqa: E402
from pydil.ipms.ot_ipms import (  # noqa: E402
    ClassConditionalWassersteinDistance
)
from pydil.dataset_distillation.distribution_matching import (  # noqa: E402
    RegularizedDistributionMatching
)

from tqdm.auto import tqdm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_distribution_matching()

if args.metric.lower() == 'mmd':
    criterion = ClassConditionalMMD()
elif args.metric.lower() == 'w2':
    criterion = ClassConditionalWassersteinDistance(reg_e=args.reg)

fname = 'caltech_office_dm.csv'
loss_name = str(criterion).replace('()', '')

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/da-distillation')
dataset = torch.from_numpy(
    np.load(os.path.join(base_path, 'data', 'Objects_Decaf7.npy'))
)

with open(os.path.join(base_path,
                       'data',
                       'Objects_Decaf_crossval_index.json')) as f:
    fold_dict = json.loads(f.read())

domain_names = ['amazon', 'caltech', 'dslr', 'webcam']

X = dataset[:, :-2].float()
y = dataset[:, -2].int()
y = y - y.min()
Y = torch.nn.functional.one_hot(y.long(), num_classes=10).float()
d = dataset[:, -1].int()

n_features = X.shape[1]
n_classes = 10
n_domains = 4

for target_domain in range(n_domains):
    names = []
    Xs, ys, Ys = [], [], []
    source_domains = [src for src in range(n_domains) if src != target_domain]
    for src in source_domains:
        ind = torch.where(d == src)[0]

        Xs_k, ys_k, Ys_k = X[ind], y[ind], Y[ind]

        μsℓ, σsℓ = Xs_k.mean(), Xs_k.std()
        Xs_k = (Xs_k - μsℓ) / σsℓ

        Xs.append(Xs_k)
        ys.append(ys_k)
        Ys.append(Ys_k)

        names.append(domain_names[src])

        ind = torch.where(d == src)[0]

    ind = torch.where(d == target_domain)[0]
    ind_tr, ind_ts = train_test_split(ind,
                                      train_size=0.8,
                                      random_state=7,
                                      stratify=y[ind])
    Xt_tr, yt_tr, Yt_tr = X[ind_tr], y[ind_tr], Y[ind_tr]
    Xt_ts, yt_ts, Yt_ts = X[ind_ts], y[ind_ts], Y[ind_ts]

    ipc_range = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    mperf, sperf = [], []
    for ipc in ipc_range:
        acc = []
        for _ in tqdm(range(5)):
            distillator = RegularizedDistributionMatching(
                spc=ipc,
                n_classes=10,
                n_dim=4096,
                xsyn=None,
                ysyn=None,
                loss_fn=criterion,
                optimizer_name=args.optimizer,
                learning_rate=args.lr,
                momentum=args.momentum,
                verbose=False
            )

            history = distillator.fit(Xs,
                                      Ys,
                                      Xt_tr,
                                      batch_size=args.batch_size,
                                      n_iter=args.n_iter,
                                      batches_per_it=10)
            xsyn = distillator.xsyn.data.clone()
            ysyn = distillator.ysyn.data.clone()
            clf = SVC(kernel='linear', max_iter=int(1e+6), C=1)
            clf.fit(xsyn, ysyn.argmax(dim=1))

            y_pred = clf.predict(Xt_ts)
            acc.append(100 * accuracy_score(y_pred, Yt_ts.argmax(dim=1)))
        m = np.mean(acc)
        s = np.std(acc)
        print(f"IPC {ipc}, perf: {m} ± {s}")

        with open(os.path.join(results_path, fname), 'a') as f:
            f.write((f'{ipc},{args.batch_size},'
                     f'{args.reg},{loss_name},{m},{s}\n'))
