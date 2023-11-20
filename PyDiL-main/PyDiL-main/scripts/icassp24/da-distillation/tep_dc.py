import os
import sys

sys.path.append('./')

import torch  # noqa: E402
import pickle  # noqa: E402
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


# Fix seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

args = parse_args_distribution_matching()

if args.metric.lower() == 'mmd':
    criterion = ClassConditionalMMD()
elif args.metric.lower() == 'w2':
    criterion = ClassConditionalWassersteinDistance(reg_e=args.reg)

fname = 'tep_dc.csv'
loss_name = str(criterion).replace('()', '')

base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = ('/home/efernand/repos/PyDiL/'
                'results/icassp24/da-distillation')
tep_path = os.path.join(
    base_path,
    'data',
    'tennessee-eastman',
    'features',
    'feature_engineering',
    'TEPFeatures.pickle'
)

domain_names = [f'Mode {k}' for k in range(1, 7)]

n_dim = 128
n_domains = 6
n_classes = 29


for target_domain in range(n_domains):
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

    fold = 0
    data = dataset[target_domain_name][f"fold {fold}"]
    Xt_tr = data['Train']['Features'].float()
    Yt_tr = data['Train']['Labels'].float()

    Xt_ts = data['Test']['Features'].float()
    Yt_ts = data['Test']['Labels'].float()

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
            distillator = RegularizedDistributionMatching(
                spc=ipc,
                n_classes=n_classes,
                n_dim=n_dim,
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
            f.write((f'{domain_names[target_domain]},{ipc},{args.batch_size},'
                     f'{args.reg},{loss_name},{m},{s}\n'))
