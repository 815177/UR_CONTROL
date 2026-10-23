"""
Alineado temporal del mocap (OptiTrack) con el robot en trayectorias ya grabadas.

En el dataset el retardo del mocap respecto a /joint_states no es constante: va de ~25 a ~290 ms
según la trayectoria y a veces salta dentro de una misma trayectoria (cliente NatNet acumulando
paquetes). El marcador de la base va rígido al gripper, así que su posición es una función conocida
de la orientación del robot: p_b(t) = c + R(phi(t - tau)) @ o. Ajustando (c, o) una vez para todo el
dataset y buscando tau por ventanas se obtiene el retardo tau(t), y se re-muestrean las columnas del
mocap en t + tau(t) para ponerlas en la línea temporal del robot.
"""
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from training.data import DLOTrajectory

MOCAP_ATTRS = ("p_b", "p_e", "dp_b", "dp_e")


def _shift(x, tau):
    """x evaluada en t + tau (tau en pasos, escalar o array por muestra), interpolación lineal"""
    t = np.arange(len(x))
    return np.stack([np.interp(t + tau, t, x[:, i]) for i in range(x.shape[1])], 1)


def _design(phi, tau):
    """Filas [I | R(phi(t - tau))] para p_b = c + R @ o (lineal en c y o)"""
    R = Rotation.from_euler("xyz", _shift(phi, -tau)).as_matrix()
    return np.concatenate([np.tile(np.eye(3), (len(phi), 1)), R.reshape(-1, 3)], 1)


def _best_lag(phi, p_b, W, taus):
    err = [((_design(phi, tau) @ W - p_b.reshape(-1)) ** 2).mean() for tau in taus]
    i = int(np.argmin(err))
    return taus[i], np.sqrt(err[i])


def fit_marker_geometry(trajs: List[DLOTrajectory], skip=30, taus=np.arange(-1, 8.01, 0.25), n_iter=2):
    """Ajusta (c, o) común a todas las trayectorias, alternando con el retardo global de cada una."""
    lags = np.zeros(len(trajs))
    W = None
    for _ in range(n_iter):
        A, b = [], []
        for k, tr in enumerate(trajs):
            phi, pb = tr.phi_b[skip:], tr.p_b[skip:]
            if W is not None:
                lags[k] = _best_lag(phi, pb, W, taus)[0]
            A.append(_design(phi, lags[k])); b.append(pb.reshape(-1))
        W, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return W


def estimate_mocap_lag(traj: DLOTrajectory, W, window=80, step=25, skip=30, taus=np.arange(-1, 8.01, 0.25)):
    """Retardo mocap -> robot (en pasos) para cada muestra, estimado por ventanas y suavizado."""
    phi, pb = traj.phi_b, traj.p_b
    centers, lags = [], []
    for s in range(skip, len(traj) - window + 1, step):
        centers.append(s + window // 2)
        lags.append(_best_lag(phi[s:s + window], pb[s:s + window], W, taus)[0])
    lags = np.array(lags)
    lags = np.array([np.median(lags[max(0, i - 1):i + 2]) for i in range(len(lags))])   # quita outliers aislados
    return np.interp(np.arange(len(traj)), centers, lags)


def align_mocap_to_robot(trajs: List[DLOTrajectory], W=None, verbose=False):
    """
    Devuelve copias de las trayectorias con p_b, p_e, dp_b, dp_e en la línea temporal del robot.
    Se recortan al final las muestras para las que no hay dato de mocap posterior.
    """
    if W is None:
        W = fit_marker_geometry(trajs)
    out = []
    for tr in trajs:
        tau = estimate_mocap_lag(tr, W)
        n_keep = len(tr) - int(np.ceil(max(tau.max(), 0))) - 1
        new = tr.get_sliced_copy(np.arange(n_keep))
        for attr in MOCAP_ATTRS:
            setattr(new, attr, _shift(getattr(tr, attr), tau)[:n_keep])
        new.mocap_lag = tau[:n_keep, None]        # (N, 1) como el resto de columnas
        out.append(new)
        if verbose:
            print(f"traj {tr.n_traj}: retardo mocap {tau.min() * 50:.0f}-{tau.max() * 50:.0f} ms (dt=50 ms)")
    return out
