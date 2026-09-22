from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

K = 16
MAX_ITER = 30


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=42)

    A = imread('data/peppers-small.tiff')

    plt.imshow(A)

    A = A.astype(float)

    m, _, n = A.shape

    mu = rng.choice(A.reshape(-1, n), size=K, axis=0, replace=False)

    c = rng.integers(0,256, (m,m))

    it = 0

    mu_prev = np.zeros((K, 3))

    while it <= MAX_ITER and not np.allclose(mu, mu_prev, atol=0.5):
        mu_prev = mu.copy()

        c_dist = np.sum((A[:,:,None,:]-mu[None,None,:,:])**2, axis=-1)
        c = np.argmin(c_dist, axis=-1)

        for k in range(K):
            mask = c==k
            if mask.any():
                mu[k] = (mask[:,:,None]*A).sum(axis=(0,1)) / mask.sum()

        it += 1

    print(it, mu-mu_prev)

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel("R")

    ax.set_ylabel("G")

    ax.set_zlabel("B")

    ax.scatter(A[:,:,0], A[:,:,1], A[:,:,2], s=1, c=c*256/K)

    plt.show()

    Ae = imread('data/peppers-small.tiff')

    plt.imshow(Ae)

    ce_dist = np.sum((Ae[:,:,None,:]-mu[None,None,:,:])**2, axis=-1)

    ce = np.argmin(c_dist, axis=-1)

    Aee = mu[ce].astype(int)

    plt.imshow(Aee)

    plt.show()
