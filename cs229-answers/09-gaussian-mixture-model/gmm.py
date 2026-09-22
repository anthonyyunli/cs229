import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']
K = 4
NUM_TRIALS = 3
UNLABELED = -1


def load_gmm_dataset(csv_path):
    """Load features and cluster labels; -1 marks an unlabeled point."""
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Save a scatter plot of cluster assignments."""
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.plot()
    plt.savefig(save_path)


def run_em(x, w, phi, mu, sigma):
    """Return cluster responsibilities from unsupervised EM."""
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        m, n = x.shape

        # E-step: responsibilities.
        for k in range(K):
            x_dot = x-mu[k]
            w[:, k] = 1 / np.sqrt(np.linalg.det(sigma[k])) * np.exp(-0.5 * (x_dot @ np.linalg.inv(sigma[k]) * x_dot).sum(axis=1)) * phi[k]
        w /= w.sum(axis=1, keepdims=True)

        # M-step: mixture weights, means, and covariances.
        phi = w.mean(axis=0)
        mu = (w.T @ x) / w.sum(axis=0)[:,None]
        for k in range(K):
            x_dot = x-mu[k]
            sigma[k] = x_dot.T @ (w[:, k][:,None] * x_dot) / w[:, k].sum()

        # Log likelihood for the convergence check.
        it += 1
        prev_ll = ll
        ll=0
        p_xz = np.zeros((m,K))
        for k in range(K):
            x_dot = x-mu[k]
            p_xz[:, k] = 1 / np.sqrt(np.linalg.det(sigma[k])) * np.exp(-0.5 * (x_dot @ np.linalg.inv(sigma[k]) * x_dot).sum(axis=1)) * phi[k]
        ll = np.sum(np.log(p_xz.sum(axis=1)))
    print(f'Number of iterations:{it}, Log loss: {ll}')

    return w


def main(is_semi_supervised, trial_num):
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    train_path = os.path.join('data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    m, n = x.shape
    idx = np.random.permutation(m)
    mu = np.ones((K,n))
    sigma = np.ones((K,n,n))

    k_size = m//K
    for k in range(K):
        start = k * k_size
        end = (k+1) * k_size if k!=K-1 else None
        x_k = x[idx[start : end]]

        mu[k] = np.average(x_k)
        sigma[k] = np.cov(x_k, rowvar=False)

    phi = np.ones((K,)) / K
    w = np.ones((m,K)) / K

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    z_pred = np.zeros(m)
    if w is not None:
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Run EM with extra weight on labeled examples."""
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        m, n = x.shape
        m_tilde, _ = x_tilde.shape

        # E-step: responsibilities.
        for k in range(K):
            x_dot = x-mu[k]
            w[:, k] = 1 / np.sqrt(np.linalg.det(sigma[k])) * np.exp(-0.5 * (x_dot @ np.linalg.inv(sigma[k]) * x_dot).sum(axis=1)) * phi[k]
        w /= w.sum(axis=1, keepdims=True)

        # M-step: mixture weights, means, and covariances.
        for k in range(K):
            mask = z.squeeze() == k
            phi[k] = (w[:, k].sum() + alpha * mask.sum()) / (m + alpha * m_tilde)
            mu[k] = (w[:, k].dot(x) + alpha * x_tilde[mask].sum(axis=0)) / (w[:, k].sum() + alpha * mask.sum())

            x_dot = x-mu[k]
            x_tilde_k = (x_tilde-mu[k])[mask]
            nk = w[:, k].sum() + alpha * mask.sum()
            sk = x_dot.T @ (w[:, k][:,None] * x_dot) + alpha * x_tilde_k.T @ x_tilde_k
            sigma[k] = sk / nk

        # Log likelihood for the convergence check.
        it += 1
        prev_ll = ll
        ll=0

        p_xz = np.zeros((m,K))
        for k in range(K):
            x_dot = x-mu[k]
            p_xz[:, k] = 1 / np.sqrt(np.linalg.det(sigma[k])) * np.exp(-0.5 * (x_dot @ np.linalg.inv(sigma[k]) * x_dot).sum(axis=1)) * phi[k]

        ll = np.sum(np.log(p_xz.sum(axis=1)))

        for k in range(K):
            mask = z.squeeze()==k
            x_dot = x_tilde[mask] - mu[k]

            p = 1 / np.sqrt(np.linalg.det(sigma[k])) * np.exp(-0.5 * (x_dot @ np.linalg.inv(sigma[k]) * x_dot).sum(axis=1)) * phi[k]
            ll += alpha * np.log(p).sum()

    print(f'Number of iterations:{it}, Log loss: {ll}')

    return w


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    np.random.seed(229)

    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)

    plt.show()
