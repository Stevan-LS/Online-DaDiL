import os
import sys

sys.path.append('./')

import json  # noqa: E402
import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

from pydil.ot_utils.barycenters import (  # noqa: E402
    wasserstein_barycenter
)
from pydil.ipms.ot_ipms import (  # noqa: E402
    JointWassersteinDistance
)
from pydil.utils.parsing import parse_args_fed_dadil  # noqa: E402
from pydil.federated_learning.client import DaDiLClient  # noqa: E402
from pydil.federated_learning.server import DaDiLServer  # noqa: E402
from pydil.torch_utils.lightning_models import ShallowNeuralNet  # noqa: E402

from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_fed_dadil()

criterion = JointWassersteinDistance()
loss_name = str(criterion).replace('()', '')

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = os.path.abspath('./results/icassp24/federated-learning/')
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
n_components = args.n_components
n_iter = args.n_iter
batch_size = args.batch_size
n_samples = args.n_samples
optimizer = args.optimizer
batches_per_it = n_samples // batch_size
n_client_it = args.n_client_it
aggregation = args.aggregation
lr = args.lr

l2_penalty = 1e-2
lr_perceptron = 1e-3
n_epochs_perceptron = 100
batch_size_perceptron = 128
optimizer_perceptron = 'sgd'


params = [
    # Optimization Parameters
    lr, n_iter, optimizer,
    # Federated Learning parameters
    n_client_it, aggregation,
    # Dictionary Parameters
    n_samples, batch_size, n_components
]

for target_domain in range(n_domains):
    target_name = domain_names[target_domain]
    source_domains = [src for src in range(n_domains) if src != target_domain]

    for trial in range(5):
        names = []
        Xs, ys, Ys = [], [], []
        clients = []
        for src in source_domains:
            ind = torch.where(d == src)[0]
            Xs_k, ys_k, Ys_k = X[ind], y[ind], Y[ind]
            μsℓ, σsℓ = Xs_k.mean(), Xs_k.std()
            Xs_k = (Xs_k - μsℓ) / σsℓ
            Xs.append(Xs_k)
            ys.append(ys_k)
            Ys.append(Ys_k)
            names.append(domain_names[src])
            clients.append(
                DaDiLClient(features=Xs_k.float(),
                            labels=Ys_k.float(),
                            n_components=n_components,
                            n_classes=n_classes,
                            lr=lr,
                            criterion=criterion,
                            batches_per_it=batches_per_it,
                            balanced_sampling=True,
                            grad_labels=True,
                            optimizer_name=optimizer)
            )
        ind = torch.where(d == target_domain)[0]
        ind_tr, ind_ts = train_test_split(ind,
                                          train_size=0.8,
                                          stratify=y[ind])
        Xt_tr, yt_tr, Yt_tr = X[ind_tr], y[ind_tr], Y[ind_tr]
        Xt_ts, yt_ts, Yt_ts = X[ind_ts], y[ind_ts], Y[ind_ts]
        μt, σt = Xt_tr.mean(), Xt_tr.std()
        Xt_tr = (Xt_tr - μt) / σt
        Xt_ts = (Xt_ts - μt) / σt
        clients.append(
            DaDiLClient(features=Xt_tr.float(),
                        n_components=n_components,
                        n_classes=n_classes,
                        lr=lr,
                        criterion=criterion,
                        batches_per_it=batches_per_it,
                        balanced_sampling=True,
                        grad_labels=True,
                        optimizer_name=optimizer)
        )
        server = DaDiLServer(
            n_samples=n_samples,
            n_dim=n_features,
            n_classes=n_classes,
            n_components=n_components,
            balanced_sampling=True,
            aggregation=aggregation
        )
        server.federated_fit(
            clients,
            spc=args.batch_size // n_classes,
            n_iter=n_iter,
            n_client_it=n_client_it,
            verbose=True
        )

        weights = clients[-1].weights.detach()

        XP = [XPk.clone() for XPk in server.XP]
        YP = [YPk.clone().softmax(dim=-1) for YPk in server.YP]
        XB, YB = wasserstein_barycenter(
            XP=XP, YP=YP,
            n_samples=n_samples,
            weights=weights,
            tol=1e-9,
            n_iter_max=10,
            propagate_labels=True,
            penalize_labels=True
        )
        model = ShallowNeuralNet(
            n_features=n_features,
            n_classes=n_classes,
            learning_rate=lr_perceptron,
            loss_fn=None,
            l2_penalty=l2_penalty,
            momentum=0.9,
            optimizer_name=optimizer_perceptron,
            log_gradients=False,
            max_norm=None
        )
        train_dataset = torch.utils.data.TensorDataset(XB, YB)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size_perceptron,
            shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(Xt_ts, Yt_ts)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=128,
                                                      shuffle=False)
        trainer = pl.Trainer(
            max_epochs=n_epochs_perceptron,
            accelerator='gpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True)
        trainer.fit(model, train_dataloader, test_dataloader)
        with torch.no_grad():
            yp_va = model(Xt_tr).argmax(dim=1)
            yp_ts = model(Xt_ts).argmax(dim=1)
            va_r_acc = 100 * accuracy_score(yp_va, yt_tr)
            ts_r_acc = 100 * accuracy_score(yp_ts, yt_ts)
            print(f"DaDiL-R got {va_r_acc}% (va), {ts_r_acc}% (ts)")

        for k, (XPk, YPk) in enumerate(zip(XP, YP)):
            train_dataset = torch.utils.data.TensorDataset(XPk, YPk)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size_perceptron,
                shuffle=True)
            model = ShallowNeuralNet(
                n_features=n_features,
                n_classes=n_classes,
                learning_rate=lr_perceptron,
                loss_fn=None,
                l2_penalty=l2_penalty,
                momentum=0.9,
                optimizer_name=optimizer_perceptron,
                log_gradients=False,
                max_norm=None
            )
            trainer = pl.Trainer(
                max_epochs=100,
                accelerator='gpu',
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True)
            trainer.fit(model, train_dataloader)

            va_predictions, ts_predictions = [], []

            # Computes statistics
            with torch.no_grad():
                # Gets predictions
                va_predictions_k = model(Xt_tr)
                ts_predictions_k = model(Xt_ts)

                va_predictions.append(va_predictions_k)
                ts_predictions.append(ts_predictions_k)

                preds_va = va_predictions_k.argmax(dim=1).cpu()
                preds_ts = ts_predictions_k.argmax(dim=1).cpu()

                atom_va_acc_k = accuracy_score(preds_va, Yt_tr.argmax(dim=1))
                atom_ts_acc_k = accuracy_score(preds_ts, Yt_ts.argmax(dim=1))

                print(f"Atom {k + 1} out of {len(XP)}")
                print(f"... weight: {weights[k]}")
                print(f"... Performance: {atom_va_acc_k}% (va)"
                      f"/ {atom_ts_acc_k}% (ts)")

        # Stacks predictions
        va_predictions = torch.stack(va_predictions)
        ts_predictions = torch.stack(ts_predictions)

        # Weights atomic model predictions
        dadil_e_va_predictions = torch.einsum('i,inj->nj',
                                              weights,
                                              va_predictions).argmax(dim=1)
        dadil_e_ts_predictions = torch.einsum('i,inj->nj',
                                              weights,
                                              ts_predictions).argmax(dim=1)
        # Compute statistics
        va_e_acc = 100 * accuracy_score(dadil_e_va_predictions,
                                        Yt_tr.argmax(dim=1)).item()
        ts_e_acc = 100 * accuracy_score(dadil_e_ts_predictions,
                                        Yt_ts.argmax(dim=1)).item()
        print(f"... DaDiL-E got {va_e_acc}% (va)"
              f"/ {ts_e_acc}% (ts) of accuracy")

        line = f"{target_name},{trial},"
        for p in params:
            line += f"{p},"
        with open(os.path.join(results_path, 'caltech_office_fed_dadil_r.csv'),
                  'a') as f:
            f.write(line + f"{va_r_acc},{ts_r_acc}\n")
        with open(os.path.join(results_path, 'caltech_office_fed_dadil_e.csv'),
                  'a') as f:
            f.write(line + f"{va_e_acc},{ts_e_acc}\n")
