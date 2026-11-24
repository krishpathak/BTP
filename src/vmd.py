import numpy as np


def vmd(signal, alpha=2000.0, tau=0.0, K=4, DC=False, init=1, tol=1e-7,
        max_iter=500):
    x = np.asarray(signal, dtype=float).ravel()
    N0 = x.size

    # mirror the signal at both ends to reduce edge effects
    ltm = N0 // 2
    f = np.concatenate((x[:ltm][::-1], x, x[-(N0 - ltm):][::-1]))
    T = f.size

    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - (1.0 / T)

    f_hat = np.fft.fftshift(np.fft.fft(f))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: T // 2] = 0.0

    Alpha = np.full(K, float(alpha))

    omega = np.zeros(K)
    if init == 1:
        omega = 0.5 / K * np.arange(K)
    elif init == 2:
        rng = np.random.default_rng(0)
        omega = np.sort(np.exp(np.log(1.0 / T) +
                               (np.log(0.5) - np.log(1.0 / T)) * rng.random(K)))
    if DC:
        omega[0] = 0.0

    u_hat_plus = np.zeros((K, T), dtype=complex)
    lamda_hat = np.zeros(T, dtype=complex)

    uDiff = tol + np.spacing(1.0)
    n = 0
    sum_uk = np.zeros(T, dtype=complex)

    while uDiff > tol and n < max_iter:
        u_hat_prev = u_hat_plus.copy()

        for k in range(K):
            sum_uk = u_hat_plus.sum(axis=0) - u_hat_plus[k]

            u_hat_plus[k] = ((f_hat_plus - sum_uk - lamda_hat / 2.0) /
                             (1.0 + Alpha[k] * (freqs - omega[k]) ** 2))

            if not (DC and k == 0):
                half = u_hat_plus[k, T // 2:]
                power = np.abs(half) ** 2
                denom = power.sum()
                if denom > 0:
                    omega[k] = (freqs[T // 2:] @ power) / denom

        lamda_hat = lamda_hat + tau * (u_hat_plus.sum(axis=0) - f_hat_plus)

        n += 1
        num = np.abs(u_hat_plus - u_hat_prev) ** 2
        den = np.abs(u_hat_prev) ** 2
        uDiff = float(num.sum() / (den.sum() + np.spacing(1.0)))

    order = np.argsort(omega)
    omega = omega[order]
    u_hat_plus = u_hat_plus[order]

    u_hat = np.zeros((K, T), dtype=complex)
    u_hat[:, T // 2:] = u_hat_plus[:, T // 2:]
    u_hat[:, 1:T // 2 + 1] = np.conj(u_hat_plus[:, T:T // 2 - 1:-1])
    u_hat[:, 0] = np.conj(u_hat[:, -1])

    u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=-1), axis=-1))
    u = u[:, ltm:ltm + N0]
    return u, u_hat, omega


def vmd_modes(signal, alpha=2000.0, tau=0.0, K=4, **kw):
    return vmd(signal, alpha=alpha, tau=tau, K=K, **kw)[0]


def vmd_batch(signals, alpha=2000.0, tau=0.0, K=4, DC=False, init=1,
              tol=1e-7, max_iter=500):
    # same algorithm as vmd() but run on a whole stack of beats at once
    x = np.atleast_2d(np.asarray(signals, dtype=float))
    B, N0 = x.shape

    ltm = N0 // 2
    f = np.concatenate((x[:, :ltm][:, ::-1], x, x[:, -(N0 - ltm):][:, ::-1]),
                       axis=1)
    T = f.shape[1]

    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - (1.0 / T)

    f_hat_plus = np.fft.fftshift(np.fft.fft(f, axis=-1), axes=-1)
    f_hat_plus[:, : T // 2] = 0.0

    if init == 1:
        omega = np.tile(0.5 / K * np.arange(K), (B, 1))
    elif init == 2:
        rng = np.random.default_rng(0)
        omega = np.sort(np.exp(np.log(1.0 / T) +
                               (np.log(0.5) - np.log(1.0 / T)) *
                               rng.random((B, K))), axis=1)
    else:
        omega = np.zeros((B, K))
    if DC:
        omega[:, 0] = 0.0

    U = np.zeros((K, B, T), dtype=complex)
    total = np.zeros((B, T), dtype=complex)
    lamda = np.zeros((B, T), dtype=complex)
    omega = omega.T.copy()

    out_U = np.zeros((K, B, T), dtype=complex)
    out_omega = np.zeros((K, B))

    pos = freqs[T // 2:]
    half = T // 2
    use_lambda = tau != 0.0
    eps = np.spacing(1.0)

    idx = np.arange(B)
    f_act = f_hat_plus
    settled = np.zeros(B, dtype=bool)
    compact_at = 0.25
    n_iter = 0

    while idx.size and n_iter < int(max_iter):
        n_iter += 1
        num = np.zeros(idx.size)
        den = np.zeros(idx.size)
        any_settled = bool(settled.any())

        for k in range(K):
            old = U[k]
            residual = f_act - total + old
            if use_lambda:
                residual = residual - lamda / 2.0
            new = residual / (1.0 + alpha *
                              (freqs[None, :] - omega[k][:, None]) ** 2)
            if any_settled:
                # keep converged beats frozen so results don't depend on batch mates
                new[settled] = old[settled]

            delta = new - old
            num += np.einsum("ij,ij->i", delta.real, delta.real) + \
                np.einsum("ij,ij->i", delta.imag, delta.imag)
            den += np.einsum("ij,ij->i", old.real, old.real) + \
                np.einsum("ij,ij->i", old.imag, old.imag)

            total += delta
            U[k] = new

            if not (DC and k == 0):
                p = new[:, half:]
                power = p.real ** 2 + p.imag ** 2
                d = power.sum(axis=1)
                good = d > 0
                if good.any():
                    omega[k][good] = (power[good] @ pos) / d[good]

        if use_lambda:
            lamda += tau * (total - f_act)

        # drop converged beats once enough of them are done
        settled |= (num / (den + eps)) <= tol
        if settled.all() or settled.sum() >= compact_at * idx.size:
            done = settled
            out_U[:, idx[done], :] = U[:, done, :]
            out_omega[:, idx[done]] = omega[:, done]
            keep = ~done
            idx = idx[keep]
            U = np.ascontiguousarray(U[:, keep, :])
            total = np.ascontiguousarray(total[keep])
            omega = np.ascontiguousarray(omega[:, keep])
            f_act = np.ascontiguousarray(f_act[keep])
            if use_lambda:
                lamda = np.ascontiguousarray(lamda[keep])
            settled = np.zeros(idx.size, dtype=bool)

    if idx.size:
        out_U[:, idx, :] = U
        out_omega[:, idx] = omega

    u_hat = np.moveaxis(out_U, 0, 1)
    omega = out_omega.T

    order = np.argsort(omega, axis=1)
    omega = np.take_along_axis(omega, order, axis=1)
    u_hat = np.take_along_axis(u_hat, order[:, :, None], axis=1)

    full = np.zeros((B, K, T), dtype=complex)
    full[:, :, T // 2:] = u_hat[:, :, T // 2:]
    full[:, :, 1:T // 2 + 1] = np.conj(u_hat[:, :, T:T // 2 - 1:-1])
    full[:, :, 0] = np.conj(full[:, :, -1])

    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=-1), axis=-1))
    return u[:, :, ltm:ltm + N0], omega, n_iter
