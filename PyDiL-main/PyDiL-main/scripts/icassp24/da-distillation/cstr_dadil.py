import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402

from pydil.utils.parsing import parse_args_dadil  # noqa: E402
from pydil.ipms.mmd_ipms import ClassConditionalMMD  # noqa: E402
from pydil.ipms.ot_ipms import (  # noqa: E402
    ClassConditionalWassersteinDistance,
    JointWassersteinDistance
)
from pydil.dadil.labeled_dictionary import LabeledDictionary  # noqa: E402
from pydil.ot_utils.barycenters import wasserstein_barycenter  # noqa: E402
from pydil.torch_utils.measures import (  # noqa: E402
    UnsupervisedDatasetMeasure,
    SupervisedDatasetMeasure
)

from scipy.fft import fft  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_dadil()

if args.metric.lower() == 'mmd':
    criterion = ClassConditionalMMD()
elif args.metric.lower() == 'w2':
    criterion = ClassConditionalWassersteinDistance(reg_e=args.reg)
elif args.metric.lower() == 'joint_w2':
    criterion = JointWassersteinDistance(reg_e=args.reg)

fname = 'cstr_dadil.csv'
loss_name = str(criterion).replace('()', '')

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
    names = []
    Xs, ys, Ys = [], [], []
    source_domains = [src for src in range(n_domains) if src != target_domain]
    Q = []
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
        Q.append(
            SupervisedDatasetMeasure(
                features=Xs_k.numpy(),
                labels=ys_k.numpy(),
                stratify=True,
                batch_size=args.batch_size,
                device='cpu'
            )
        )

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

    Q.append(
        UnsupervisedDatasetMeasure(
            features=Xt_tr.numpy(),
            batch_size=args.batch_size,
            device='cpu'
        )
    )

    dictionary = LabeledDictionary(XP=None,
                                   YP=None,
                                   A=None,
                                   n_samples=args.n_samples,
                                   n_dim=n_dim,
                                   n_classes=n_classes,
                                   n_components=args.n_components,
                                   weight_initialization='uniform',
                                   n_distributions=len(Q),
                                   loss_fn=criterion,
                                   learning_rate_features=args.lr,
                                   learning_rate_labels=args.lr,
                                   learning_rate_weights=args.lr,
                                   reg_e=args.reg,
                                   n_iter_barycenter=10,
                                   n_iter_sinkhorn=20,
                                   n_iter_emd=1000000,
                                   domain_names=None,
                                   grad_labels=True,
                                   optimizer_name=args.optimizer,
                                   balanced_sampling=True,
                                   sampling_with_replacement=True,
                                   barycenter_tol=1e-9,
                                   barycenter_beta=None,
                                   tensor_dtype=torch.float32,
                                   track_atoms=False,
                                   schedule_lr=False)
    dictionary.fit(Q,
                   n_iter_max=args.n_iter,
                   batches_per_it=args.n_samples // args.batch_size,
                   verbose=True)

    XP = [XPk.data.clone() for XPk in dictionary.XP]
    YP = [YPk.data.clone().softmax(dim=1) for YPk in dictionary.YP]
    A = dictionary.A.data.clone()

    accs = []
    accs_err = []

    ipc_range = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    mperf, sperf = [], []
    for ipc in ipc_range:
        acc = []
        for _ in tqdm(range(5)):
            XB, YB = wasserstein_barycenter(
                XP=XP,
                YP=YP,
                XB=None,
                YB=None,
                weights=A[-1, :],
                n_samples=ipc * n_classes,
                reg_e=0.0,
                label_weight=None,
                n_iter_max=10,
                tol=1e-9,
                verbose=False,
                propagate_labels=True,
                penalize_labels=True
            )

            clf = SVC(kernel='linear', max_iter=int(1e+6), C=1)
            clf.fit(XB, YB.argmax(dim=1))

            y_pred = clf.predict(Xt_ts)
            acc.append(100 * accuracy_score(y_pred, Yt_ts.argmax(dim=1)))
        m = np.mean(acc)
        s = np.std(acc)

        accs.append(m)
        accs_err.append(s)

        print(f"IPC {ipc}, perf: {m} ± {s}")

        with open(os.path.join(results_path, fname), 'a') as f:
            f.write((f'{target_domain},{ipc},{args.n_samples},'
                     f'{args.batch_size},{args.n_components},'
                     f'{args.reg},{loss_name},{m},{s}\n'))
