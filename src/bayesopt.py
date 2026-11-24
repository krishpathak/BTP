import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class Space:
    # spec maps name -> ("real" | "log" | "int", lo, hi) or ("cat", [choices])
    def __init__(self, spec):
        self.spec = dict(spec)
        self.names = list(self.spec)

    @property
    def ndim(self):
        return len(self.names)

    def to_params(self, u):
        out = {}
        for ui, name in zip(np.clip(u, 0.0, 1.0), self.names):
            kind, *rest = self.spec[name]
            if kind == "cat":
                choices = rest[0]
                out[name] = choices[min(int(ui * len(choices)), len(choices) - 1)]
            elif kind == "int":
                lo, hi = rest
                out[name] = int(round(lo + ui * (hi - lo)))
            elif kind == "log":
                lo, hi = rest
                out[name] = float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))))
            else:
                lo, hi = rest
                out[name] = float(lo + ui * (hi - lo))
        return out


def _expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def minimize(objective, space, n_iter=30, n_init=8, n_candidates=2000,
             random_state=0, verbose=True):
    rng = np.random.default_rng(random_state)
    d = space.ndim

    X = rng.random((n_init, d))
    y, history = [], []
    for u in X:
        p = space.to_params(u)
        v = float(objective(p))
        y.append(v)
        history.append((p, v))
        if verbose:
            print(f"  [init] {v:.6f}  {p}", flush=True)
    y = np.asarray(y)

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              Matern(length_scale=np.ones(d), nu=2.5,
                     length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(1e-6, (1e-10, 1e-1)))

    for it in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2,
                                      random_state=random_state)
        gp.fit(X, y)

        cand = rng.random((n_candidates, d))
        mu, sd = gp.predict(cand, return_std=True)
        ei = _expected_improvement(mu, sd, y.min())
        u = cand[int(np.argmax(ei))]

        p = space.to_params(u)
        v = float(objective(p))
        X = np.vstack([X, u])
        y = np.append(y, v)
        history.append((p, v))
        if verbose:
            print(f"  [{it + 1:2d}/{n_iter}] {v:.6f}  (best {y.min():.6f})  {p}",
                  flush=True)

    best = int(np.argmin(y))
    return space.to_params(X[best]), float(y[best]), history
