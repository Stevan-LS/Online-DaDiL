import os
import sys

sys.path.append('./')

import json  # noqa: E402
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

if args.multi:
    fname = 'caltech_office_multi_domain_dadil.csv'
else:
    fname = 'caltech_office_dadil.csv'
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
                                   n_dim=4096,
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
            barycenters = []

            XB0 = torch.randn(ipc * 10, 4096)
            yB0 = torch.cat([
                torch.Tensor([c] * ipc) for c in range(10)]).long()
            YB0 = torch.nn.functional.one_hot(
                yB0, num_classes=10).float()

            XB, YB = wasserstein_barycenter(
                XP=XP,
                YP=YP,
                XB=None,
                YB=None,
                weights=A[-1, :],
                n_samples=ipc * 10,
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
            f.write((f'{domain_names[target_domain]},{ipc},{args.n_samples},'
                     f'{args.batch_size},{args.n_components},'
                     f'{args.reg},{loss_name},{m},{s}\n'))
