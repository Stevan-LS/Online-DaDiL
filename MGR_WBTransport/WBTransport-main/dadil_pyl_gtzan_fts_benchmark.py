import os
import sys

sys.path.append(os.path.abspath('./'))

import json
import torch
import warnings
import argparse
import numpy as np
import tensorflow as tf
import pytorch_lightning as pl

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from dictionary_learning.utils import BalancedBatchSamplerDA
from dictionary_learning.losses import JointWassersteinLoss
from dictionary_learning.lightning_dictionary import LightningDictionary
from dictionary_learning.utils import DictionaryDADataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Arguments for Wasserstein Dictionary Learning')
parser.add_argument('--base_path',
                    type=str,
                    default="./")
parser.add_argument('--n_samples',
                    type=int,
                    default=2000)
parser.add_argument('--batch_size',
                    type=int,
                    default=130)
parser.add_argument('--reg_e',
                    type=float,
                    default=0.0)
parser.add_argument('--reg_e_dictionary',
                    type=float,
                    default=0.0)
parser.add_argument('--reg_m',
                    type=float,
                    default=0.0)
parser.add_argument('--reg_A',
                    type=float,
                    default=0.0)
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-1)
parser.add_argument('--n_components',
                    type=int,
                    default=8)
parser.add_argument('--device',
                    type=str,
                    default='cuda')
parser.add_argument('--num_iter_barycenter',
                    type=int,
                    default=1)
parser.add_argument('--num_iter_sinkhorn',
                    type=int,
                    default=50)
parser.add_argument('--num_iter_dil',
                    type=int,
                    default=10)
parser.add_argument('--batches_per_it',
                    type=int,
                    default=10)
parser.add_argument('--feature_type',
                    type=str,
                    default='decaf')
parser.add_argument('--penalize_labels',
                    type=str,
                    default='true')
parser.add_argument('--stratify',
                    type=str,
                    default='true')
parser.add_argument('--log',
                    type=str,
                    default='true')
args = parser.parse_args()

# Parameters
# ----------
base_path = os.path.abspath('./data')
n_samples = args.n_samples
batch_size = args.batch_size
n_components = args.n_components
ϵ = args.reg_e
ϵ_dictionary = args.reg_e_dictionary
τ = args.reg_m
η_A = args.reg_A
lr = args.learning_rate
num_iter_barycenter = args.num_iter_barycenter
num_iter_sinkhorn = args.num_iter_sinkhorn
num_iter_dil = args.num_iter_dil
batches_per_it = n_samples // batch_size
device = args.device
penalize_labels = True if args.penalize_labels.lower() == 'true' else False
stratify = True if args.stratify.lower() == 'true' else False
log = True if args.log.lower() == 'true' else False
n_classes = 10

dataset = np.load(os.path.join(base_path, 'MGR.npy'))
with open(os.path.join(base_path, 'MGR_crossval_index.json'), 'r') as f:
    fold_dict = json.loads(f.read())
clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)

# Features
X = dataset[:, :-2]
X = np.delete(X, 17, axis=1)
# Feature Scaling
# X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Class and domain labels
y = dataset[:, -2]
m = dataset[:, -1]

n_dim = n_features = X.shape[1]

# clf = RandomForestClassifier(n_estimators=1000, max_depth=13, n_jobs=-1, random_state=0)
clf = SVC(kernel='linear', C=10)

domains = ['original', 'factory2', 'f16', 'destroyerengine', 'buccaneer2']

for target_name in domains[1:]:
    target = [d for d in range(len(domains)) if domains[d] == target_name][0]
    sources = [d for d in range(len(domains)) if d != target]
    accs = []
    for i in [1, 2, 3, 4, 5]:
        selected_folds = [j for j in range(1, 6) if j != i]
        unselected_fold = i
        inds = [
            np.concatenate([
                fold_dict['Domain {}'.format(s + 1)]['Fold {}'.format(f)] for f in selected_folds
            ]) for s in sources
        ]

        indt = np.concatenate([
            fold_dict['Domain {}'.format(target + 1)]['Fold {}'.format(f)] for f in selected_folds
        ])
        indt_ts = fold_dict['Domain {}'.format(target + 1)]['Fold {}'.format(i)]
    

        Xs, ys = [], []

        for ind in inds:
            _xs = torch.from_numpy(X[ind]).double()
            _ys = torch.from_numpy(y[ind]).double()
            
            _xs = (_xs - _xs.mean(dim=0)) / _xs.std(dim=0)

            Xs.append(_xs)
            ys.append(_ys)

        Ys = [torch.nn.functional.one_hot(ys_k.long(), num_classes=n_classes).double() for ys_k in ys]

        cXs = np.concatenate(Xs, axis=0)
        cys = np.concatenate(ys, axis=0)

        Xt = torch.from_numpy(X[indt]).double()
        Xt = (Xt - Xt.mean(dim=0)) / Xt.std(dim=0)
        yt = torch.from_numpy(y[indt]).double()

        # Fitting baseline
        clf.fit(cXs, cys)
        yp = clf.predict(Xt)
        acc_baseline = accuracy_score(y_pred=yp, y_true=yt)
        print("[{}] Baseline: {}".format(target_name, 100 * acc_baseline))

        train_dataset = DictionaryDADataset(Xs, Ys, Xt, None, None)
        S = BalancedBatchSamplerDA([Ysk.argmax(dim=1).numpy() for Ysk in Ys],
                                   n_target=len(Xt),
                                   n_classes=n_classes,
                                   batch_size=batch_size,
                                   n_batches=batches_per_it)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=S)

        names = [domains[s] for s in sources] + [domains[target]]

        all_Xs = torch.cat(Xs, dim=0)
        all_ys = torch.cat(Ys, dim=0).argmax(dim=1)
        XP = [[] for _ in range(n_components)]
        YP = [[] for _ in range(n_components)]
        for c in range(n_classes):
            ind = torch.where(all_ys == c)[0]
            mean_c = all_Xs[ind].mean(dim=0)
            for k in range(n_components):
                XP[k].append(mean_c + torch.randn((n_samples // n_classes), n_dim))
                YP[k].append(torch.nn.functional.one_hot(torch.tensor([c] * (n_samples // n_classes)), num_classes=n_classes))
        for k in range(n_components):
            XP[k] = torch.cat(XP[k], dim=0).to(Xs[0].dtype)
            YP[k] = torch.cat(YP[k], dim=0).to(Xs[0].dtype)

        # Creates dictionary
        loss_fn = JointWassersteinLoss(ϵ=1e-3, num_iter_sinkhorn=100)
        dictionary = LightningDictionary(XP=XP, YP=YP,
                                         n_samples=n_samples,
                                         n_dim=n_features,
                                         n_classes=n_classes,
                                         n_components=n_components,
                                         n_distributions=len(Xs) + 1,
                                         learning_rate_features=5e-2,
                                         learning_rate_labels=5e-2,
                                         learning_rate_weights=5e-2,
                                         reg=0.0,
                                         reg_labels=0.0,
                                         loss_fn=loss_fn,
                                         domain_names=names,
                                         proj_grad=True,
                                         grad_labels=False,
                                         pseudo_label=False,
                                         balanced_sampling=True,
                                         sampling_with_replacement=True,
                                         weight_initialization='uniform',
                                         num_iter_barycenter=100,
                                         num_iter_sinkhorn=100,
                                         barycenter_initialization='class',
                                         barycenter_covariance_type='diag',
                                         barycenter_verbose=False,
                                         barycenter_tol=1e-9,
                                         batch_size=batch_size,
                                         dtype='double')

        # Creates logger
        if log:
            logger = TensorBoardLogger(save_dir=os.path.abspath('./results/gtzan/logs/multi-source/dadil/Dictionary/'),
                                        name="{}".format(domains[target]),
                                        log_graph=True)
        else:
            logger = False

        # Creates trainer object
        trainer = pl.Trainer(max_epochs=num_iter_dil, accelerator='cpu', logger=logger, enable_checkpointing=False)
        trainer.fit(dictionary, train_loader)

        weights = dictionary.A[-1, :].detach().cpu()

        # Reconstruct samples
        Xr, Yr = dictionary.reconstruct(α=weights, n_samples_atoms=None, n_samples_barycenter=None)
        Xr = Xr.detach().cpu()
        Yr = Yr.detach().cpu()
        yr = Yr.argmax(dim=1)

        clf.fit(Xr, yr)
        yp = clf.predict(Xt)
        dadil_r_acc = accuracy_score(yt, yp)

        # Prints results
        print(r'[{}] DaDiL-R: {}'.format(target_name, 100 * dadil_r_acc))

        w = dictionary.A[-1].cpu().detach()
        predictions = []
        for k, (XP_k, YP_k) in enumerate(zip(dictionary.XP, dictionary.YP)):
            # Get atom data
            XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
            yp_k = YP_k.argmax(dim=1)
            
            clf.fit(XP_k, yp_k)
            yp = clf.predict(Xt)
            print("Atom {} w/ weight {} got {}".format(k, np.round(w[k], 3), accuracy_score(yt, yp)))

            P = clf.predict_proba(Xt)
            predictions.append(P)
        predictions = np.stack(predictions)
        
        # Weights atomic model predictions
        yp = np.einsum('i,inj->nj', w, predictions).argmax(axis=1)

        # Compute statistics
        dadil_e_acc = accuracy_score(yt, yp)

        # Prints results
        print(r'[{}] DaDiL-E: {}'.format(target_name, 100 * dadil_e_acc))

        with open(os.path.abspath(os.path.abspath('./results/gtzan/fts_dadil.csv')), 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(domains[target], n_components,
                                                 n_samples, batch_size,
                                                 dadil_r_acc, dadil_e_acc))