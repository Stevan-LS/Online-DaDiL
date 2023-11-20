import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import pickle  # noqa: E402
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
loss_name = str(criterion).replace('()', '')

fname = 'cwru_dadil.csv'

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

    Q, Xs, Ys = [], [], []
    for source in sources:
        # Gets features and labels from matrices
        Xsk = dataset[source]['Features']
        Ysk = dataset[source]['Labels'].float()

        # Standardizes features
        # Xs = (Xs - Xs.mean()) / (Xs.std())
        print('... Source {}: {}'.format(source, Xsk.shape))

        Q.append(
            SupervisedDatasetMeasure(
                features=Xsk.numpy(),
                labels=Ysk.argmax(dim=1).numpy(),
                stratify=True,
                batch_size=args.batch_size,
                device='cpu'
            )
        )

        Xs.append(Xsk)
        Ys.append(Ysk)
    Xall = torch.cat(Xs, dim=0)
    Yall = torch.cat(Ys, dim=0)
    Xt_tr = dataset[target_domain]['fold 0']['Train']["Features"]
    Yt_tr = dataset[target_domain]['fold 0']['Train']["Labels"].float()
    Xt_ts = dataset[target_domain]['fold 0']['Test']["Features"]
    Yt_ts = dataset[target_domain]['fold 0']['Test']["Labels"].float()
    # Xt = (Xt - Xt.mean()) / Xt.std()
    print('... Target {}'.format(target_domain))
    print('...... Train: {}'.format(Xt_tr.shape))
    print('...... Test:  {}'.format(Xt_ts.shape))

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
