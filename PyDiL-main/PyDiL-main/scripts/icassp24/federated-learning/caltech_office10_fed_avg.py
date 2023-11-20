import os
import json
import torch
import numpy as np

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def model_avg(state_dicts, device='cpu'):
    with torch.no_grad():
        _w = state_dicts[0].copy()
        keys = state_dicts[0].keys()

        with torch.no_grad():
            for k in keys:
                param_versions = []
                for state in state_dicts:
                    param_versions.append(state[k].data)
                _w[k].data = torch.stack(param_versions).mean(dim=0).to(device)
    return _w


def apply_device_on_state_dict(state_dict, device='cpu'):
    return {k: v.to(device) for k, v in state_dict.items()}


base_path = '/home/efernand/repos/dataset-dictionary-learning'
results_path = os.path.abspath('./results/icassp24/federated-learning/')
fname = 'caltech_office10_fed_avg.csv'
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
batch_size = 128
n_client_it = 10

device = 'cuda'

lr = 1e-3
l2_penalty = 1e-2
batch_size = 128


# Number of local iterations
E = 5

# Number of Rounds of Communication
R = 10

for target_domain in range(n_domains):
    target_name = domain_names[target_domain]
    source_domains = [src for src in range(n_domains) if src != target_domain]

    Xs, Ys = [], []
    clients = []
    for src in source_domains:
        ind = torch.where(d == src)[0]
        Xs_k, Ys_k = X[ind], Y[ind]
        μsℓ, σsℓ = Xs_k.mean(), Xs_k.std()
        Xs_k = (Xs_k - μsℓ) / σsℓ
        Xs.append(Xs_k)
        Ys.append(Ys_k)

    client_data = []
    for Xsk, Ysk in zip(Xs, Ys):
        client_data.append(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(Xsk, Ysk),
                batch_size=batch_size,
                shuffle=True
            )
        )

    model = torch.nn.Sequential(torch.nn.Linear(4096, 10)).to(device)
    w0 = apply_device_on_state_dict(
        model.state_dict(), device='cpu').copy()

    client_models = [
        torch.nn.Sequential(
            torch.nn.Linear(4096, 10)
        ).to(device) for _ in range(len(Xs))]

    for h in client_models:
        h.load_state_dict(
            apply_device_on_state_dict(w0.copy(), device=device)
        )

    client_history = [[] for _ in range(len(Xs))]
    server_history = {'loss': [], 'acc': []}

    criterion = torch.nn.CrossEntropyLoss()

    w = w0.copy()

    for r in range(R):
        print(f'Round {r}')
        for nc, (dl, h, history) in enumerate(
                zip(client_data, client_models, client_history)):
            optimizer = torch.optim.SGD(h.parameters(), lr=lr,
                                        weight_decay=l2_penalty,
                                        momentum=0.9)

            for e in range(E):
                it_loss = 0.0
                for x, y in tqdm(dl):
                    x = x.to(device)
                    y = y.argmax(dim=1).to(device)

                    # Zeroes grads
                    optimizer.zero_grad()

                    # Predicts
                    yhat = h(x)

                    # loss
                    loss = criterion(yhat, target=y)

                    # MMD
                    # loss += .1 * mmd(z, zt, beta=20)

                    # backprop
                    loss.backward()

                    # grad step
                    optimizer.step()

                    it_loss += loss.item() / len(dl)
                history.append(it_loss)
                print(f'client {nc}, epoch {e}, loss {it_loss}')

        # Model aggregation
        client_state_dicts = [
            apply_device_on_state_dict(h.state_dict(), device='cpu').copy()
            for h in client_models]
        avg_state_dict = model_avg(client_state_dicts, device=device)

        # Model synchronization
        with torch.no_grad():
            for h in client_models:
                h.load_state_dict(avg_state_dict)

    for h in client_models:
        h.to('cpu')

    for f in range(5):
        ind = torch.where(d == target_domain)[0]
        ind_tr, ind_ts = train_test_split(ind,
                                          train_size=0.8,
                                          stratify=Y.argmax(dim=1)[ind])
        Xt_tr, Yt_tr = X[ind_tr], Y[ind_tr]
        Xt_ts, Yt_ts = X[ind_ts], Y[ind_ts]
        μt, σt = Xt_tr.mean(), Xt_tr.std()
        Xt_tr = (Xt_tr - μt) / σt
        Xt_ts = (Xt_ts - μt) / σt

        with torch.no_grad():
            yp_va = client_models[0](Xt_tr).argmax(dim=1)
            yp_ts = client_models[0](Xt_ts).argmax(dim=1)

            va_acc = accuracy_score(yp_va, Yt_tr.argmax(dim=1))
            ts_acc = accuracy_score(yp_ts, Yt_ts.argmax(dim=1))

            print(domain_names[target_domain], va_acc, ts_acc)

        with open(os.path.join(results_path, fname), 'a') as f:
            f.write(f'{domain_names[target_domain]},{va_acc},{ts_acc}\n')
