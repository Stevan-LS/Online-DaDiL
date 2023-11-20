import os
import sys

sys.path.append('./')

import ot  # noqa: E402
import umap  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

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
from pydil.dataset_distillation.distribution_matching import (  # noqa: E402
    RegularizedDistributionMatching
)

from sklearn.svm import SVC  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
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

n_samples = 500
batch_size = 200
n_components = 3

target_domain = 0

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

ipc = 1
Xdadil, Ydadil = wasserstein_barycenter(
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

Xwbt, Ywbt = wasserstein_barycenter(
    XP=Xs,
    YP=Ys,
    XB=None,
    YB=None,
    weights=None,
    n_samples=ipc * 10,
    reg_e=0.0,
    label_weight=None,
    n_iter_max=10,
    tol=1e-9,
    verbose=False,
    propagate_labels=True,
    penalize_labels=True
)

Xwbt = ot.da.EMDTransport().fit_transform(
    Xs=Xwbt, ys=None, Xt=Xt_tr, yt=None
)

distillator = RegularizedDistributionMatching(
    spc=ipc,
    n_classes=10,
    n_dim=4096,
    xsyn=None,
    ysyn=None,
    loss_fn=ClassConditionalMMD(),
    optimizer_name='sgd',
    learning_rate=1,
    momentum=0.9,
    verbose=True
)

history = distillator.fit(Xs, Ys, Xt_tr,
                          batch_size=250,
                          n_iter=30,
                          batches_per_it=10)

Xdc = distillator.xsyn.detach().clone()
Ydc = distillator.ysyn.detach().clone()

X_for_tsne = torch.cat(
    [Xt_tr, Xt_ts, Xdadil, Xwbt, Xdc],
    dim=0
)

offset = 0
cutting_pts = []
for arr in [Xt_tr, Xt_ts, Xdadil, Xwbt, Xdc]:
    cutting_pts.append(offset)
    offset += len(arr)

y_for_tsne = torch.cat(
    [Yt_tr, Yt_ts, Ydadil, Ywbt, Ydc],
    dim=0
).argmax(dim=1)

norms = torch.linalg.norm(X_for_tsne, dim=1) ** 2
inner_prods = torch.mm(X_for_tsne, X_for_tsne.T)
dists = 1 - ((inner_prods) / (torch.sqrt(norms[:, None] * norms[None, :])))
dists[dists < 0.0] = 0.0
proj = TSNE(n_components=2, metric='precomputed').fit_transform(dists)
proj = umap.UMAP(
    n_neighbors=50, min_dist=1, metric='cosine').fit_transform(X_for_tsne)

plt.figure(figsize=(5, 5))
plt.scatter(proj[:, 0], proj[:, 1], c=y_for_tsne, cmap='tab10', alpha=0.1)
start, end = cutting_pts[2], cutting_pts[3]
plt.scatter(proj[start:end, 0],
            proj[start:end, 1],
            c=y_for_tsne[start:end],
            cmap='tab10',
            marker='*',
            label='DaDiL',
            edgecolor='k')
start, end = cutting_pts[3], cutting_pts[4]
plt.scatter(proj[start:end, 0],
            proj[start:end, 1],
            c=y_for_tsne[start:end],
            cmap='tab10',
            marker='^',
            label='WBT',
            edgecolor='k')
start, end = cutting_pts[4], 988
plt.scatter(proj[start:end, 0],
            proj[start:end, 1],
            c=y_for_tsne[start:end],
            cmap='tab10',
            marker='P',
            label='MSDA-DC',
            edgecolor='k')
plt.legend(loc='lower center', ncols=3,
           bbox_to_anchor=(0.5, 1.0))
plt.axis('off')
plt.tight_layout()
plt.show()

clf_dc = SVC(kernel='linear', max_iter=1e+6, C=1)
clf_dc.fit(Xdc, Ydc.argmax(dim=1))
yp = clf_dc.predict(Xt_ts)
cmat_dc = confusion_matrix(y_true=yt_ts, y_pred=yp)
print("[DC] acc: {}".format(accuracy_score(yp, yt_ts)))

clf_dadil = SVC(kernel='linear', max_iter=1e+6, C=1)
clf_dadil.fit(Xdadil, Ydadil.argmax(dim=1))
yp = clf_dadil.predict(Xt_ts)
cmat_dadil = confusion_matrix(y_true=yt_ts, y_pred=yp)
print("[DaDiL] acc: {}".format(accuracy_score(yp, yt_ts)))

clf_wbt = SVC(kernel='linear', max_iter=1e+6, C=1)
clf_wbt.fit(Xwbt, Ywbt.argmax(dim=1))
yp = clf_wbt.predict(Xt_ts)
cmat_wbt = confusion_matrix(y_true=yt_ts, y_pred=yp)
print("[WBT] acc: {}".format(accuracy_score(yp, yt_ts)))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
sns.heatmap(cmat_dc.astype(int), ax=axes[0], annot=True, cbar=False)
sns.heatmap(cmat_dadil.astype(int), ax=axes[1], annot=True, cbar=False)
sns.heatmap(cmat_wbt.astype(int), ax=axes[2], annot=True)

axes[0].set_title('MSDA-DC')
axes[1].set_title('DaDiL')
axes[2].set_title('WBT')

axes[0].set_ylabel('Grount-Truth')
axes[0].set_xlabel('Predicted')
axes[1].set_xlabel('Predicted')
axes[2].set_xlabel('Predicted')

plt.show()
