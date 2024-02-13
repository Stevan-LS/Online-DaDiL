import torch  # noqa: E402
import random  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt
import ot
import os
import pickle

from pydil.utils.igmm_modif import IGMM

from pydil.ipms.ot_ipms import (  # noqa: E402
    JointWassersteinDistance
)
from pydil.dadil.labeled_dictionary_GMM import LabeledDictionaryGMM
from pydil.torch_utils.measures import (  # noqa: E402
    UnsupervisedDatasetMeasure,
    SupervisedDatasetMeasure
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def prepare_dataset_cwru(target, fold):
    with open(os.path.join('data', f'mlp_fts_256_target_{target}.pkl'), 'rb') as f:
            dataset = pickle.loads(f.read())

    Xs, ys = [], []
    d = None
    keys = list(dataset.keys())
    for i in [0, 1]:
        features = dataset[keys[i]]['Features']
        labels = dataset[keys[i]]['Labels'].argmax(dim=1)
        domain = i*np.ones((features.shape[0], 1))
        Xs.append(features.float())
        ys.append(labels.float())
        if d is None:
            d = domain
        else:
            d = np.concatenate([d, domain], axis=0)

    Xt = dataset[target][f'fold {fold}']['Train']['Features'].float()
    yt = dataset[target][f'fold {fold}']['Train']['Labels'].float().argmax(dim=1)

    Xt_test = dataset[target][f'fold {fold}']['Test']['Features'].float()
    yt_test = dataset[target][f'fold {fold}']['Test']['Labels'].float().argmax(dim=1)
    d = np.concatenate([d, 2*np.ones((Xt.shape[0], 1))], axis=0)

    n_domains = int(np.max(d)) + 1
    n_features = Xt.shape[1]
    n_classes = int(np.max(yt.numpy())) + 1

    return Xs, ys, Xt, yt, Xt_test, yt_test, n_features

def test_dadil_cwru_all_folds(target, n_samples, n_classes, n_atoms, batch_size, n_iter):
    results = {'rbf':{'wda': [], 'e':[], 'e_ot':[], 'r':[], 'r_ot':[]}}
    n_folds = 5
    for j in range(n_folds):
        print(f'{j+1}/{n_folds}')
        Xs, ys, Xt, yt, Xt_test, yt_test, n_features = prepare_dataset_cwru(target, j)

        Q = []
        for Xs_k, ys_k in zip(Xs, ys):
            Q.append(
                SupervisedDatasetMeasure(
                    features=Xs_k.numpy(),
                    labels=ys_k.numpy(),
                    stratify=True,
                    batch_size=batch_size,
                    device='cpu'
                )
            )
        Q.append(
            UnsupervisedDatasetMeasure(
                features=Xt.numpy(),
                batch_size=batch_size,
                device='cpu'
            )
        )
        criterion = JointWassersteinDistance()
        dictionary = LabeledDictionaryGMM(XP=None,
                                YP=None,
                                A=None,
                                n_samples=n_samples,
                                n_dim=n_features,
                                n_classes=n_classes,
                                n_components=n_atoms,
                                weight_initialization='uniform',
                                n_distributions=len(Q),
                                loss_fn=criterion,
                                learning_rate_features=1e-1,
                                learning_rate_labels=1e-1,
                                learning_rate_weights=1e-1,
                                reg_e=0.0,
                                n_iter_barycenter=10,
                                n_iter_sinkhorn=20,
                                n_iter_emd=1000000,
                                domain_names=None,
                                grad_labels=True,
                                optimizer_name='Adam',
                                balanced_sampling=True,
                                sampling_with_replacement=True,
                                barycenter_tol=1e-9,
                                barycenter_beta=None,
                                tensor_dtype=torch.float32,
                                track_atoms=False,
                                schedule_lr=False)
        dictionary.fit(Q,
                    n_iter_max=n_iter,
                    batches_per_it=n_samples // batch_size,
                    verbose=True)
        weights = dictionary.A[-1, :].detach()
        XP = [XPk.detach().clone() for XPk in dictionary.XP]
        YP = [YPk.detach().clone().softmax(dim=-1) for YPk in dictionary.YP]
        Xr, Yr = dictionary.reconstruct(weights=weights)

        classifiers_e = {'rbf': SVC(kernel='rbf', probability=True)}
        classifiers_r = {'rbf': SVC(kernel='rbf',)}

        
        for key in classifiers_e.keys():
            # Without DA
            print('wda')
            clf_wda = classifiers_r[key]
            clf_wda.fit(torch.cat(Xs, dim=0),
                    torch.cat(ys, dim=0))
            yp = clf_wda.predict(Xt_test)
            accuracy_wda = accuracy_score(yp, yt_test)
            results[key]['wda'].append(accuracy_wda)

            # DaDiL-E
            print('e')
            clf_e = classifiers_e[key]
            predictions = []
            for XP_k, YP_k in zip(XP, YP):
                # Get atom data
                XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
                yp_k = YP_k.argmax(dim=1)
                clf_e.fit(XP_k, yp_k)
                P = clf_e.predict_proba(Xt_test)
                predictions.append(P)
            predictions = np.stack(predictions)
            # Weights atomic model predictions
            yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
            # Compute statistics
            accuracy_e = accuracy_score(yt_test, yp)
            results[key]['e'].append(accuracy_e)

            # DaDiL-E with last optimal transport
            print('e ot')
            s = 0
            for _ in range(10):
                predictions = []
                for XP_k, YP_k in zip(XP, YP):
                    # Get atom data
                    XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
                    weights_k = torch.ones(XP_k.shape[0])/XP_k.shape[0]
                    weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
                    C = torch.cdist(XP_k, Xt, p=2) ** 2
                    ot_plan = ot.emd(weights_k, weights_t, C, numItermax=1000000)
                    Yt = ot_plan.T @ YP_k
                    yt_k = Yt.argmax(dim=1)
                    clf_e.fit(Xt, yt_k)
                    P = clf_e.predict_proba(Xt_test)
                    predictions.append(P)
                predictions = np.stack(predictions)
                # Weights atomic model predictions
                yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
                # Compute statistics
                accuracy_e_ot = accuracy_score(yt_test, yp)
                s += accuracy_e_ot
            mean_accuracy_e_ot = s/10
            results[key]['e_ot'].append(mean_accuracy_e_ot)

            # DaDiL-R
            print('r')
            clf_r = classifiers_r[key]
            clf_r.fit(Xr, Yr.argmax(dim=1))
            yp = clf_r.predict(Xt_test)
            accuracy_r = accuracy_score(yp, yt_test)
            results[key]['r'].append(accuracy_r)

            # DaDiL-R with last optimal transport
            print('r ot')
            s = 0
            for _ in range(10):
                weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
                weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
                C = torch.cdist(Xr, Xt, p=2) ** 2
                ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ Yr
                clf_r.fit(Xt, Yt.argmax(dim=1))
                yp = clf_r.predict(Xt_test)
                accuracy_r_ot = accuracy_score(yp, yt_test)
                s += accuracy_r_ot
            results[key]['r_ot'].append(s/10)
    
    for kr in results.keys():
        for kk in results[kr].keys():
            L = results[kr][kk]
            results[kr][kk] = (np.mean(L), np.std(L))
    
    return results, dictionary

def test_odadil_cwru_fold(target, fold, n_samples, n_classes, n_atoms, batch_size, n_iter):
    results = {'rbf':{'wda': 0, 'e':0, 'e_ot':0, 'r':0, 'r_ot':0}}

    Xs, ys, Xt, yt, Xt_test, yt_test, n_features = prepare_dataset_cwru(target, fold)

    Q_sources = []
    for Xs_k, ys_k in zip(Xs, ys):
        Q_sources.append(
            SupervisedDatasetMeasure(
                features=Xs_k.numpy(),
                labels=ys_k.numpy(),
                stratify=True,
                batch_size=batch_size,
                device='cpu'
            )
        )

    criterion = JointWassersteinDistance()

    dictionary_sources = LabeledDictionaryGMM(XP=None,
                            YP=None,
                            A=None,
                            n_samples=n_samples,
                            n_dim=n_features,
                            n_classes=n_classes,
                            n_components=n_atoms,
                            weight_initialization='uniform',
                            n_distributions=len(Q_sources),
                            loss_fn=criterion,
                            learning_rate_features=1e-1,
                            learning_rate_labels=1e-1,
                            learning_rate_weights=1e-1,
                            reg_e=0.0,
                            n_iter_barycenter=10,
                            n_iter_sinkhorn=20,
                            n_iter_emd=1000000,
                            domain_names=None,
                            grad_labels=True,
                            optimizer_name='Adam',
                            balanced_sampling=True,
                            sampling_with_replacement=True,
                            barycenter_tol=1e-9,
                            barycenter_beta=None,
                            tensor_dtype=torch.float32,
                            track_atoms=False,
                            schedule_lr=False)

    dictionary_sources.fit(Q_sources,
                n_iter_max=n_iter,
                batches_per_it=n_samples // batch_size,
                verbose=True)

    XP_sources = dictionary_sources.XP
    YP_sources = dictionary_sources.YP

    dictionary_target = LabeledDictionaryGMM(XP=XP_sources,
                                    YP=YP_sources,
                                    A=None,
                                    n_samples=n_samples,
                                    n_dim=n_features,
                                    n_classes=n_classes,
                                    n_components=n_atoms,
                                    weight_initialization='uniform',
                                    n_distributions=1,
                                    loss_fn=criterion,
                                    learning_rate_features=1e-2,
                                    learning_rate_labels=1e-2,
                                    learning_rate_weights=1e-1,
                                    reg_e=0.0,
                                    n_iter_barycenter=10,
                                    n_iter_sinkhorn=20,
                                    n_iter_emd=1000000,
                                    domain_names=None,
                                    grad_labels=True,
                                    optimizer_name='Adam',
                                    balanced_sampling=True,
                                    sampling_with_replacement=True,
                                    barycenter_tol=1e-9,
                                    barycenter_beta=None,
                                    tensor_dtype=torch.float32,
                                    track_atoms=False,
                                    schedule_lr=False,
                                    min_components=10,
                                    max_step_components=10,
                                    max_components=20)
    
    n_batch = 20
    i = 0
    while i < Xt.shape[0]-n_batch:
        dictionary_target.fit_target_sample(Xt[i:i+n_batch, :],
                                            batches_per_it=n_samples // batch_size,
                                            batch_size=batch_size,
                                            verbose=True,
                                            regularization=False,)
        print(f'{i}/{Xt.shape[0]}')
        i += n_batch

    weights = dictionary_target.A[-1, :].detach()
    XP = [XPk.detach().clone() for XPk in dictionary_target.XP]
    YP = [YPk.detach().clone().softmax(dim=-1) for YPk in dictionary_target.YP]

    Xr, Yr = dictionary_target.reconstruct(weights=weights)

    classifiers_e = {'rbf': SVC(kernel='rbf', probability=True)}
    classifiers_r = {'rbf': SVC(kernel='rbf')}

    for key in classifiers_e.keys():
        # Without DA
        print('wda')
        clf_wda = classifiers_r[key]
        clf_wda.fit(torch.cat(Xs, dim=0),
                torch.cat(ys, dim=0))
        yp = clf_wda.predict(Xt_test)
        accuracy_wda = accuracy_score(yp, yt_test)
        results[key]['wda'] = accuracy_wda

        #DaDiL-E
        print('e')
        clf_e = classifiers_e[key]
        predictions = []
        for XP_k, YP_k in zip(XP, YP):
            # Get atom data
            XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
            yp_k = YP_k.argmax(dim=1)
            clf_e.fit(XP_k, yp_k)
            P = clf_e.predict_proba(Xt_test)
            predictions.append(P)
        predictions = np.stack(predictions)
        # Weights atomic model predictions
        yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
        # Compute statistics
        accuracy_e = accuracy_score(yt_test, yp)
        results[key]['e'] = accuracy_e

        #DaDiL-E with last optimal transport
        print('e_ot')
        s = 0
        for _ in range(10):
            predictions = []
            for XP_k, YP_k in zip(XP, YP):
                # Get atom data
                XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
                weights_k = torch.ones(XP_k.shape[0])/XP_k.shape[0]
                weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
                C = torch.cdist(XP_k, Xt, p=2) ** 2
                ot_plan = ot.emd(weights_k, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ YP_k
                yt_k = Yt.argmax(dim=1)
                clf_e.fit(Xt, yt_k)
                P = clf_e.predict_proba(Xt_test)
                predictions.append(P)
            predictions = np.stack(predictions)
            # Weights atomic model predictions
            yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
            # Compute statistics
            accuracy_e_ot = accuracy_score(yt_test, yp)
            s += accuracy_e_ot
        results[key]['e_ot'] = s/10

        #DaDiL-R
        print('r')
        clf_r = classifiers_r[key]
        clf_r.fit(Xr, Yr.argmax(dim=1))
        yp = clf_r.predict(Xt)
        accuracy_r = accuracy_score(yp, yt)
        results[key]['r'] = accuracy_r

        #DaDiL-R with last optimal transport
        print('r_ot')
        s = 0
        for _ in range(10):
            weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
            weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
            C = torch.cdist(Xr, Xt, p=2) ** 2
            ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
            Yt = ot_plan.T @ Yr
            clf_r.fit(Xt, Yt.argmax(dim=1))
            yp = clf_r.predict(Xt_test)
            accuracy_r_ot = accuracy_score(yp, yt_test)
            s += accuracy_r_ot
        results[key]['r_ot'] = s/10
    
    return results, dictionary_sources, dictionary_target

def test_dadil(Xs, ys, Xt, yt, Xt_test, yt_test, n_features, n_samples, n_classes, n_atoms, batch_size, n_iter):
    results = {'lin':{'wda': 0, 'e':0, 'e_ot':0, 'r':0, 'r_ot':0}, 'rbf':{'wda': 0, 'e':0, 'e_ot':0, 'r':0, 'r_ot':0}}

    Q = []
    for Xs_k, ys_k in zip(Xs, ys):
        Q.append(
            SupervisedDatasetMeasure(
                features=Xs_k.numpy(),
                labels=ys_k.numpy(),
                stratify=True,
                batch_size=batch_size,
                device='cpu'
            )
        )
    Q.append(
        UnsupervisedDatasetMeasure(
            features=Xt.numpy(),
            batch_size=batch_size,
            device='cpu'
        )
    )
    criterion = JointWassersteinDistance()
    dictionary = LabeledDictionaryGMM(XP=None,
                            YP=None,
                            A=None,
                            n_samples=n_samples,
                            n_dim=n_features,
                            n_classes=n_classes,
                            n_components=n_atoms,
                            weight_initialization='uniform',
                            n_distributions=len(Q),
                            loss_fn=criterion,
                            learning_rate_features=1e-1,
                            learning_rate_labels=1e-1,
                            learning_rate_weights=1e-1,
                            reg_e=0.0,
                            n_iter_barycenter=10,
                            n_iter_sinkhorn=20,
                            n_iter_emd=1000000,
                            domain_names=None,
                            grad_labels=True,
                            optimizer_name='Adam',
                            balanced_sampling=True,
                            sampling_with_replacement=True,
                            barycenter_tol=1e-9,
                            barycenter_beta=None,
                            tensor_dtype=torch.float32,
                            track_atoms=False,
                            schedule_lr=False)
    dictionary.fit(Q,
                n_iter_max=n_iter,
                batches_per_it=n_samples // batch_size,
                verbose=True)
    weights = dictionary.A[-1, :].detach()
    XP = [XPk.detach().clone() for XPk in dictionary.XP]
    YP = [YPk.detach().clone().softmax(dim=-1) for YPk in dictionary.YP]
    Xr, Yr = dictionary.reconstruct(weights=weights)

    classifiers_e = {'lin': SVC(kernel='linear', probability=True), 'rbf': SVC(kernel='rbf', probability=True)}
    classifiers_r = {'lin': SVC(kernel='linear'), 'rbf': SVC(kernel='rbf')}

    
    for key in classifiers_e.keys():
        # Without DA
        clf_wda = classifiers_r[key]
        clf_wda.fit(torch.cat(Xs, dim=0),
                torch.cat(ys, dim=0))
        yp = clf_wda.predict(Xt_test)
        accuracy_wda = accuracy_score(yp, yt_test)
        results[key]['wda'] += accuracy_wda

        # DaDiL-E
        clf_e = classifiers_e[key]
        predictions = []
        for XP_k, YP_k in zip(XP, YP):
            # Get atom data
            XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
            yp_k = YP_k.argmax(dim=1)
            clf_e.fit(XP_k, yp_k)
            P = clf_e.predict_proba(Xt_test)
            predictions.append(P)
        predictions = np.stack(predictions)
        # Weights atomic model predictions
        yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
        # Compute statistics
        accuracy_e = accuracy_score(yt_test, yp)
        results[key]['e'] += accuracy_e

        # DaDiL-E with last optimal transport
        s = 0
        for _ in range(10):
            predictions = []
            for XP_k, YP_k in zip(XP, YP):
                # Get atom data
                XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
                weights_k = torch.ones(XP_k.shape[0])/XP_k.shape[0]
                weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
                C = torch.cdist(XP_k, Xt, p=2) ** 2
                ot_plan = ot.emd(weights_k, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ YP_k
                yt_k = Yt.argmax(dim=1)
                clf_e.fit(Xt, yt_k)
                P = clf_e.predict_proba(Xt_test)
                predictions.append(P)
            predictions = np.stack(predictions)
            # Weights atomic model predictions
            yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
            # Compute statistics
            accuracy_e_ot = accuracy_score(yt_test, yp)
            s += accuracy_e_ot
        mean_accuracy_e_ot = s/10
        results[key]['e_ot'] += mean_accuracy_e_ot

        # DaDiL-R
        clf_r = classifiers_r[key]
        clf_r.fit(Xr, Yr.argmax(dim=1))
        yp = clf_r.predict(Xt_test)
        accuracy_r = accuracy_score(yp, yt_test)
        results[key]['r'] += accuracy_r

        # DaDiL-R with last optimal transport
        s = 0
        for _ in range(10):
            weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
            weights_t = torch.ones(Xt.shape[0])/Xt.shape[0]
            C = torch.cdist(Xr, Xt, p=2) ** 2
            ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
            Yt = ot_plan.T @ Yr
            clf_r.fit(Xt, Yt.argmax(dim=1))
            yp = clf_r.predict(Xt_test)
            accuracy_r_ot = accuracy_score(yp, yt_test)
            s += accuracy_r_ot
        results[key]['r_ot'] += s/10

    
    return results


def test_odadil(Xs, ys, Xt, yt, Xt_test, yt_test, n_features, n_samples, n_classes, n_atoms, batch_size, n_iter):
    results = {'lin':{'wda': 0, 'e':0, 'e_ot':0, 'r':0, 'r_ot':0}, 'rbf':{'wda': 0, 'e':0, 'e_ot':0, 'r':0, 'r_ot':0}}
    
    Q_sources = []
    for Xs_k, ys_k in zip(Xs, ys):
        Q_sources.append(
            SupervisedDatasetMeasure(
                features=Xs_k.numpy(),
                labels=ys_k.numpy(),
                stratify=True,
                batch_size=batch_size,
                device='cpu'
            )
        )

    criterion = JointWassersteinDistance()

    dictionary_sources = LabeledDictionaryGMM(XP=None,
                            YP=None,
                            A=None,
                            n_samples=n_samples,
                            n_dim=n_features,
                            n_classes=n_classes,
                            n_components=n_atoms,
                            weight_initialization='uniform',
                            n_distributions=len(Q_sources),
                            loss_fn=criterion,
                            learning_rate_features=1e-1,
                            learning_rate_labels=1e-1,
                            learning_rate_weights=1e-1,
                            reg_e=0.0,
                            n_iter_barycenter=10,
                            n_iter_sinkhorn=20,
                            n_iter_emd=1000000,
                            domain_names=None,
                            grad_labels=True,
                            optimizer_name='Adam',
                            balanced_sampling=True,
                            sampling_with_replacement=True,
                            barycenter_tol=1e-9,
                            barycenter_beta=None,
                            tensor_dtype=torch.float32,
                            track_atoms=False,
                            schedule_lr=False)

    dictionary_sources.fit(Q_sources,
                n_iter_max=n_iter,
                batches_per_it=n_samples // batch_size,
                verbose=True)

    XP_sources = dictionary_sources.XP
    YP_sources = dictionary_sources.YP

    dictionary_target = LabeledDictionaryGMM(XP=XP_sources,
                                    YP=YP_sources,
                                    A=None,
                                    n_samples=n_samples,
                                    n_dim=n_features,
                                    n_classes=n_classes,
                                    n_components=n_atoms,
                                    weight_initialization='uniform',
                                    n_distributions=1,
                                    loss_fn=criterion,
                                    learning_rate_features=1e-2,
                                    learning_rate_labels=1e-2,
                                    learning_rate_weights=1e-1,
                                    reg_e=0.0,
                                    n_iter_barycenter=10,
                                    n_iter_sinkhorn=20,
                                    n_iter_emd=1000000,
                                    domain_names=None,
                                    grad_labels=True,
                                    optimizer_name='Adam',
                                    balanced_sampling=True,
                                    sampling_with_replacement=True,
                                    barycenter_tol=1e-9,
                                    barycenter_beta=None,
                                    tensor_dtype=torch.float32,
                                    track_atoms=False,
                                    schedule_lr=False,
                                    min_components=10,
                                    max_step_components=10,
                                    max_components=20)
    
    n_batch = 20
    i = 0
    while i < Xt.shape[0]-n_batch:
        dictionary_target.fit_target_sample(Xt[i:i+n_batch, :],
                                            batches_per_it=n_samples // batch_size,
                                            batch_size=batch_size,
                                            verbose=True,
                                            regularization=False,)
        i += n_batch

    weights = dictionary_target.A[-1, :].detach()
    XP = [XPk.detach().clone() for XPk in dictionary_target.XP]
    YP = [YPk.detach().clone().softmax(dim=-1) for YPk in dictionary_target.YP]

    Xr, Yr = dictionary_target.reconstruct(weights=weights)

    classifiers_e = {'lin': SVC(kernel='linear', probability=True), 'rbf': SVC(kernel='rbf', probability=True)}
    classifiers_r = {'lin': SVC(kernel='linear'), 'rbf': SVC(kernel='rbf')}
    ogmm_samples = dictionary_target.OGMM.sample(6000)[0]

    for key in classifiers_e.keys():
        # Without DA
        clf_wda = classifiers_r[key]
        clf_wda.fit(torch.cat(Xs, dim=0),
                torch.cat(ys, dim=0))
        yp = clf_wda.predict(Xt_test)
        accuracy_wda = accuracy_score(yp, yt_test)
        results[key]['wda'] += accuracy_wda

        #DaDiL-E
        clf_e = classifiers_e[key]
        predictions = []
        for XP_k, YP_k in zip(XP, YP):
            # Get atom data
            XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
            yp_k = YP_k.argmax(dim=1)
            clf_e.fit(XP_k, yp_k)
            P = clf_e.predict_proba(Xt_test)
            predictions.append(P)
        predictions = np.stack(predictions)
        # Weights atomic model predictions
        yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
        # Compute statistics
        accuracy_e = accuracy_score(yt_test, yp)
        results[key]['e'] += accuracy_e

        #DaDiL-E with last optimal transport
        s = 0
        for _ in range(10):
            predictions = []
            for XP_k, YP_k in zip(XP, YP):
                # Get atom data
                XP_k, YP_k = XP_k.data.cpu(), YP_k.data.cpu()
                weights_k = torch.ones(XP_k.shape[0])/XP_k.shape[0]
                weights_t = torch.ones(ogmm_samples.shape[0])/ogmm_samples.shape[0]
                C = torch.cdist(XP_k, ogmm_samples, p=2) ** 2
                ot_plan = ot.emd(weights_k, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ YP_k
                yt_k = Yt.argmax(dim=1)
                clf_e.fit(ogmm_samples, yt_k)
                P = clf_e.predict_proba(Xt_test)
                predictions.append(P)
            predictions = np.stack(predictions)
            # Weights atomic model predictions
            yp = np.einsum('i,inj->nj', weights, predictions).argmax(axis=1)
            # Compute statistics
            accuracy_e_ot = accuracy_score(yt_test, yp)
            s += accuracy_e_ot
        results[key]['e_ot'] += s/10

        #DaDiL-R
        clf_r = classifiers_r[key]
        clf_r.fit(Xr, Yr.argmax(dim=1))
        yp = clf_r.predict(Xt_test)
        accuracy_r = accuracy_score(yp, yt_test)
        results[key]['r'] += accuracy_r

        #DaDiL-R with last optimal transport
        s = 0
        for _ in range(10):
            weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
            weights_t = torch.ones(ogmm_samples.shape[0])/ogmm_samples.shape[0]
            C = torch.cdist(Xr, ogmm_samples, p=2) ** 2
            ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
            Yt = ot_plan.T @ Yr
            clf_r.fit(ogmm_samples, Yt.argmax(dim=1))
            yp = clf_r.predict(Xt_test)
            accuracy_r_ot = accuracy_score(yp, yt_test)
            s += accuracy_r_ot
        results[key]['r_ot'] += s/10

    return results, dictionary_sources, dictionary_target


def test_forgetting_odadil(Xs, ys, Xt, yt, Xt_test, yt_test, n_features, n_samples, n_classes, n_atoms, batch_size, n_iter):
    before_online_results = {'lin':{'r':[], 'r_ot':[]}, 
               'rbf':{'r':[], 'r_ot':[]}}
    after_online_results = {'lin':{'r':[], 'r_ot':[]}, 
               'rbf':{'r':[], 'r_ot':[]}}
    classifiers_r = {'lin': SVC(kernel='linear'), 'rbf': SVC(kernel='rbf')}

    Q_sources = []
    for Xs_k, ys_k in zip(Xs, ys):
        Q_sources.append(
            SupervisedDatasetMeasure(
                features=Xs_k.numpy(),
                labels=ys_k.numpy(),
                stratify=True,
                batch_size=batch_size,
                device='cpu'
            )
        )

    criterion = JointWassersteinDistance()

    dictionary_sources = LabeledDictionaryGMM(XP=None,
                            YP=None,
                            A=None,
                            n_samples=n_samples,
                            n_dim=n_features,
                            n_classes=n_classes,
                            n_components=n_atoms,
                            weight_initialization='uniform',
                            n_distributions=len(Q_sources),
                            loss_fn=criterion,
                            learning_rate_features=1e-1,
                            learning_rate_labels=1e-1,
                            learning_rate_weights=1e-1,
                            reg_e=0.0,
                            n_iter_barycenter=10,
                            n_iter_sinkhorn=20,
                            n_iter_emd=1000000,
                            domain_names=None,
                            grad_labels=True,
                            optimizer_name='Adam',
                            balanced_sampling=True,
                            sampling_with_replacement=True,
                            barycenter_tol=1e-9,
                            barycenter_beta=None,
                            tensor_dtype=torch.float32,
                            track_atoms=False,
                            schedule_lr=False)

    dictionary_sources.fit(Q_sources,
                n_iter_max=n_iter,
                batches_per_it=n_samples // batch_size,
                verbose=True)

    XP_sources = dictionary_sources.XP
    YP_sources = dictionary_sources.YP

    weights_list = dictionary_sources.A.detach()

    # Test classif sources avant OGMM
    for i in range(len(weights_list)):
        Xr, Yr = dictionary_sources.reconstruct(weights=weights_list[i])

        for key in classifiers_r.keys():
            #DaDiL-R
            clf_r = classifiers_r[key]
            clf_r.fit(Xr, Yr.argmax(dim=1))
            yp = clf_r.predict(Xs[i])
            accuracy_r = accuracy_score(yp, ys[i])
            before_online_results[key]['r'].append(accuracy_r)

            #DaDiL-R with last optimal transport
            s = 0
            for _ in range(10):
                weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
                weights_t = torch.ones(Xs[i].shape[0])/Xs[i].shape[0]
                C = torch.cdist(Xr, Xs[i], p=2) ** 2
                ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ Yr
                clf_r.fit(Xs[i], Yt.argmax(dim=1))
                yp = clf_r.predict(Xs[i])
                accuracy_r_ot = accuracy_score(yp, ys[i])
                s += accuracy_r_ot
            before_online_results[key]['r_ot'].append(s/10)

    # Online Learning
    dictionary_target = LabeledDictionaryGMM(XP=XP_sources,
                                    YP=YP_sources,
                                    A=None,
                                    n_samples=n_samples,
                                    n_dim=n_features,
                                    n_classes=n_classes,
                                    n_components=n_atoms,
                                    weight_initialization='uniform',
                                    n_distributions=1,
                                    loss_fn=criterion,
                                    learning_rate_features=1e-2,
                                    learning_rate_labels=1e-2,
                                    learning_rate_weights=1e-1,
                                    reg_e=0.0,
                                    n_iter_barycenter=10,
                                    n_iter_sinkhorn=20,
                                    n_iter_emd=1000000,
                                    domain_names=None,
                                    grad_labels=True,
                                    optimizer_name='Adam',
                                    balanced_sampling=True,
                                    sampling_with_replacement=True,
                                    barycenter_tol=1e-9,
                                    barycenter_beta=None,
                                    tensor_dtype=torch.float32,
                                    track_atoms=False,
                                    schedule_lr=False,
                                    min_components=10,
                                    max_step_components=10,
                                    max_components=20)
    
    n_batch = 20
    c = 0
    while c < Xt.shape[0]-n_batch:
        dictionary_target.fit_target_sample(Xt[c:c+n_batch, :],
                                            batches_per_it=n_samples // batch_size,
                                            batch_size=batch_size,
                                            verbose=True,
                                            regularization=False,)
        c += n_batch


    # Test classif sources après online learning
    for i in range(len(weights_list)):
        Xr, Yr = dictionary_target.reconstruct(weights=weights_list[i])

        for key in classifiers_r.keys():
            #DaDiL-R
            clf_r = classifiers_r[key]
            clf_r.fit(Xr, Yr.argmax(dim=1))
            yp = clf_r.predict(Xs[i])
            accuracy_r = accuracy_score(yp, ys[i])
            after_online_results[key]['r'].append(accuracy_r)

            #DaDiL-R with last optimal transport
            s = 0
            for _ in range(10):
                weights_r = torch.ones(Xr.shape[0])/Xr.shape[0]
                weights_t = torch.ones(Xs[i].shape[0])/Xs[i].shape[0]
                C = torch.cdist(Xr, Xs[i], p=2) ** 2
                ot_plan = ot.emd(weights_r, weights_t, C, numItermax=1000000)
                Yt = ot_plan.T @ Yr
                clf_r.fit(Xs[i], Yt.argmax(dim=1))
                yp = clf_r.predict(Xs[i])
                accuracy_r_ot = accuracy_score(yp, ys[i])
                s += accuracy_r_ot
            after_online_results[key]['r_ot'].append(s/10)
    
    return before_online_results, after_online_results