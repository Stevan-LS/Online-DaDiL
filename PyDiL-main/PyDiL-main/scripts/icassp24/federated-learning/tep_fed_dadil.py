import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import pickle  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

from pydil.ipms.ot_ipms import JointWassersteinDistance  # noqa: E402
from pydil.utils.parsing import parse_args_fed_dadil  # noqa: E402
from pydil.ot_utils.barycenters import wasserstein_barycenter  # noqa: E402
from pydil.federated_learning.client import DaDiLClient  # noqa: E402
from pydil.federated_learning.server import DaDiLServer  # noqa: E402
from pydil.torch_utils.lightning_models import ShallowNeuralNet  # noqa: E402

from sklearn.metrics import accuracy_score  # noqa: E402


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_fed_dadil()
print(args)
criterion = JointWassersteinDistance(reg_e=0.0)

fname = 'tep_fed_dadil'
loss_name = str(criterion).replace('()', '')

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/federated-learning')
tep_path = os.path.join(
    base_path,
    'data',
    'tennessee-eastman',
    'features',
    'feature_engineering',
    'TEPFeatures.pickle'
)

domain_names = [f'Mode {k}' for k in range(1, 7)]

n_components = args.n_components
n_iter = args.n_iter
batch_size = args.batch_size
n_samples = args.n_samples
batches_per_it = n_samples // batch_size
n_client_it = args.n_client_it
optimizer = args.optimizer
aggregation = args.aggregation
lr = args.lr

n_dim = 128
n_domains = 6
n_classes = 29

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
    names = []

    source_domains_names = [
        "Mode {}".format(d + 1)
        for d in range(n_domains) if d != target_domain]
    target_domain_name = f"Mode {target_domain + 1}"

    data_path = os.path.join(
        base_path,
        f"data/tennessee-eastman/features/"
        f"tep_fcn_features_target_{target_domain + 1}.pkl"
    )
    with open(data_path, 'rb') as f:
        dataset = pickle.loads(f.read())

    for fold in range(5):
        Xs, Ys = [], []
        for source_domain in source_domains_names:
            Xs_k = dataset[source_domain]['Features'].float()
            Ys_k = dataset[source_domain]['Labels'].float()

            μs_k, σs_k = Xs_k.mean(), Xs_k.std()
            Xs_k = (Xs_k - μs_k) / σs_k

            ind = np.arange(len(Xs_k))
            np.random.shuffle(ind)

            Xs.append(Xs_k)
            Ys.append(Ys_k)
        n_classes = Ys_k.shape[1]
        n_features = Xs_k.shape[1]

        clients = []
        for Xs_k, Ys_k in zip(Xs, Ys):
            clients.append(
                DaDiLClient(
                    features=Xs_k.float(),
                    labels=Ys_k.float(),
                    n_components=n_components,
                    n_classes=n_classes,
                    lr=lr,
                    criterion=criterion,
                    batches_per_it=batches_per_it,
                    balanced_sampling=True,
                    grad_labels=True,
                    optimizer_name=optimizer
                )
            )

        data = dataset[target_domain_name][f"fold {fold}"]
        Xt_tr = data['Train']['Features'].float()
        Yt_tr = data['Train']['Labels'].float()

        Xt_ts = data['Test']['Features'].float()
        Yt_ts = data['Test']['Labels'].float()

        μt, σt = Xt_tr.mean(), Xt_tr.std()
        Xt_tr = (Xt_tr - μt) / σt
        Xt_ts = (Xt_ts - μt) / σt

        clients.append(
            DaDiLClient(
                features=Xt_tr.float(),
                labels=None,
                n_components=n_components,
                n_classes=n_classes,
                lr=lr,
                criterion=criterion,
                batches_per_it=batches_per_it,
                grad_labels=True,
                optimizer_name=optimizer
            )
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
            XP=XP,
            YP=YP,
            XB=None,
            YB=None,
            weights=weights,
            n_samples=n_samples,
            reg_e=0.0,
            label_weight=None,
            n_iter_max=10,
            tol=1e-9,
            verbose=False,
            propagate_labels=True,
            penalize_labels=True
        )

        model = ShallowNeuralNet(
            n_features=n_dim,
            n_classes=n_classes,
            learning_rate=1e-2,
            loss_fn=None,
            l2_penalty=1e-3,
            momentum=0.9,
            optimizer_name='sgd',
            log_gradients=False,
            max_norm=None
        )
        train_dataset = torch.utils.data.TensorDataset(XB, YB)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=128,
                                                       shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(Xt_ts, Yt_ts)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=128,
                                                      shuffle=False)
        trainer = pl.Trainer(
            max_epochs=150,
            accelerator='gpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True)
        trainer.fit(model, train_dataloader, test_dataloader)

        with torch.no_grad():
            yp_va = model(Xt_tr).argmax(dim=1)
            yp_ts = model(Xt_ts).argmax(dim=1)
            va_r_acc = 100 * accuracy_score(yp_va, Yt_tr.argmax(dim=1))
            ts_r_acc = 100 * accuracy_score(yp_ts, Yt_ts.argmax(dim=1))
            print(f"DaDiL-R got {va_r_acc}% (va), {ts_r_acc}% (ts)")

        for k, (XPk, YPk) in enumerate(zip(XP, YP)):
            train_dataset = torch.utils.data.TensorDataset(XPk, YPk)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=128,
                                                       shuffle=True)
            model = ShallowNeuralNet(
                n_features=n_dim,
                n_classes=n_classes,
                learning_rate=1e-2,
                loss_fn=None,
                l2_penalty=1e-3,
                momentum=0.9,
                optimizer_name='sgd',
                log_gradients=False,
                max_norm=None
            )
            trainer = pl.Trainer(
                max_epochs=150,
                accelerator='gpu',
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True)
            trainer.fit(model, train_loader)

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

        line = f"{target_name},{fold},"
        for p in params:
            line += f"{p},"
        with open(os.path.join(results_path, fname + '_r.csv'),
                  'a') as f:
            f.write(line + f"{va_r_acc},{ts_r_acc}\n")
        with open(os.path.join(results_path, fname + '_e.csv'),
                  'a') as f:
            f.write(line + f"{va_e_acc},{ts_e_acc}\n")
