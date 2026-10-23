import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import copy
from dataclasses import dataclass
from typing import Protocol


# ---------- Escalador tipo sklearn ----------
class TorchMinMaxScaler:
    """Scaler compatible con NumPy y PyTorch, diferenciable con tensores."""
    def __init__(self, feature_range=(-1.0, 1.0)):
        self.min, self.max = feature_range
        self.data_min = None
        self.data_max = None

    def fit(self, x):
        if isinstance(x, np.ndarray):
            self.data_min = np.min(x, axis=0)
            self.data_max = np.max(x, axis=0)
        elif isinstance(x, torch.Tensor):
            self.data_min = x.min(0).values
            self.data_max = x.max(0).values
        else:
            raise TypeError("x debe ser numpy.ndarray o torch.Tensor")

    def transform(self, x):
        if isinstance(x, np.ndarray):
            x_std = (x - self.data_min) / (self.data_max - self.data_min + 1e-8)
            return x_std * (self.max - self.min) + self.min
        elif isinstance(x, torch.Tensor):
            # Mover data_min y data_max al mismo dispositivo que x
            if not isinstance(self.data_min, torch.Tensor):
                data_min = torch.tensor(self.data_min, device=x.device, dtype=x.dtype)
                data_max = torch.tensor(self.data_max, device=x.device, dtype=x.dtype)
            else:
                data_min = self.data_min.to(x.device)
                data_max = self.data_max.to(x.device)

            x_std = (x - data_min) / (data_max - data_min + 1e-8)
            return x_std * (self.max - self.min) + self.min
        else:
            raise TypeError("x debe ser numpy.ndarray o torch.Tensor")

    def inverse_transform(self, x):
        if isinstance(x, np.ndarray):
            x_std = (x - self.min) / (self.max - self.min + 1e-8)
            return x_std * (self.data_max - self.data_min) + self.data_min
        elif isinstance(x, torch.Tensor):
            if not isinstance(self.data_min, torch.Tensor):
                data_min = torch.tensor(self.data_min, device=x.device, dtype=x.dtype)
                data_max = torch.tensor(self.data_max, device=x.device, dtype=x.dtype)
            else:
                data_min = self.data_min.to(x.device)
                data_max = self.data_max.to(x.device)

            x_std = (x - self.min) / (self.max - self.min + 1e-8)
            return x_std * (data_max - data_min) + data_min
        else:
            raise TypeError("x debe ser numpy.ndarray o torch.Tensor")

class TorchStandardScaler(TorchMinMaxScaler):
    """(x - media) / std. Misma interfaz que TorchMinMaxScaler; menos sensible a picos que min-max."""
    def __init__(self):
        super().__init__(feature_range=(0.0, 1.0))

    def fit(self, x):
        if isinstance(x, np.ndarray):
            mean, std = np.mean(x, axis=0), np.std(x, axis=0)
        elif isinstance(x, torch.Tensor):
            mean, std = x.mean(0), x.std(0)
        else:
            raise TypeError("x debe ser numpy.ndarray o torch.Tensor")
        # Reutiliza transform/inverse_transform de min-max con data_min = media, data_max - data_min = std
        self.data_min = mean
        self.data_max = mean + std


# ---------- Dataset ----------
@dataclass
class DLODataset:
    U_encoder: torch.Tensor = None
    U_dyn: torch.Tensor = None
    U_decoder: torch.Tensor = None
    Y: torch.Tensor = None
    encoder_scalar: TorchMinMaxScaler = None
    dyn_scalar: TorchMinMaxScaler = None
    output_scalar: TorchMinMaxScaler = None


class DLODatasetScalars(Protocol):
    encoder_scalar: TorchMinMaxScaler = None
    dyn_scalar: TorchMinMaxScaler = None
    output_scalar: TorchMinMaxScaler = None

from torch.utils.data import Dataset


class TorchAltDataset(Dataset):
    def __init__(self, X, Y, input_scalar=None, output_scalar=None):

        self.X = X.detach().clone() if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)
        self.Y = Y.detach().clone() if torch.is_tensor(Y) else torch.tensor(Y, dtype=torch.float32)

        self.input_scalar = input_scalar
        self.output_scalar = output_scalar

    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



class TorchDLODataset(Dataset):
    def __init__(self, U_encoder, U_dyn, U_decoder, Y,
                 encoder_scalar=None, dyn_scalar=None, output_scalar=None, decoder_scalar=None):

        self.U_encoder = U_encoder.detach().clone() if torch.is_tensor(U_encoder) else torch.tensor(U_encoder, dtype=torch.float32)
        self.U_dyn     = U_dyn.detach().clone()     if torch.is_tensor(U_dyn) else torch.tensor(U_dyn, dtype=torch.float32)
        self.U_decoder = U_decoder.detach().clone() if torch.is_tensor(U_decoder) else torch.tensor(U_decoder, dtype=torch.float32)
        self.Y         = Y.detach().clone()         if torch.is_tensor(Y) else torch.tensor(Y, dtype=torch.float32)

        self.encoder_scalar = encoder_scalar
        self.dyn_scalar     = dyn_scalar
        self.output_scalar  = output_scalar
        self.decoder_scalar = decoder_scalar

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return (self.U_encoder[idx],
                self.U_dyn[idx],
                self.U_decoder[idx],
                self.Y[idx])
    

# ---------- Trayectorias ----------
class DLOTrajectory:
    def __init__(self, traj_path: str, n_traj: int = None) -> None:
        axes = ['x', 'y', 'z']
        actions = ['pitch', 'yaw']
        self.n_traj = n_traj
        self.data = pd.read_csv(traj_path)

        self.t = self.data[['t']].values

        self.p_b_cols = [f'p_b_{x}' for x in axes]
        self.phi_b_cols = [f'phi_b_{x}' for x in axes]
        self.dp_b_cols = [f'dp_b_{x}' for x in axes]
        self.dphi_b_cols = [f'dphi_b_{x}' for x in axes]
        self.p_e_cols = [f'p_e_{x}' for x in axes]
        self.dp_e_cols = [f'dp_e_{x}' for x in axes]

        self.u_pitch_yaw_cols =  [f'u_{a}' for a in actions]
        # self.force_cols = [f'force_{x}' for x in axes]
        # self.torque_cols = [f'torque_{x}' for x in axes]

        self.p_b = self.data[self.p_b_cols].values
        self.phi_b = self.data[self.phi_b_cols].values
        self.dp_b = self.data[self.dp_b_cols].values
        self.dphi_b = self.data[self.dphi_b_cols].values
        self.p_e = self.data[self.p_e_cols].values
        self.dp_e = self.data[self.dp_e_cols].values
        
        self.u_pitch_yaw = self.data[self.u_pitch_yaw_cols].values
        # self.force = self.data[self.force_cols].values
        # self.torque = self.data[self.torque_cols].values

    def get_sliced_copy(self, idxs):
        sliced_obj = copy.copy(self)
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                if isinstance(attr, np.ndarray):
                    setattr(sliced_obj, attr_name, attr[idxs, :])
        return sliced_obj

    def __len__(self):
        return self.t.shape[0]

    def plot(self, variable):
        if isinstance(variable, list):
            vars_, lbls = [], []
            for v in variable:
                if hasattr(self, v):
                    vars_.append(getattr(self, v))
                    lbls.append(getattr(self, v + '_cols'))
            plot_trajs([self.t] * len(vars_), vars_, lbls)
        else:
            if hasattr(self, variable):
                plot_trajs(self.t, getattr(self, variable), getattr(self, variable + '_cols'))


# ---------- Dataloader ----------
class DLODataLoader:
    def __init__(self, U_enc: torch.Tensor, U_dyn: torch.Tensor,
                 U_dec: torch.Tensor, Y: torch.Tensor, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.U_enc = U_enc
        self.U_dec = U_dec
        self.U_dyn = U_dyn
        self.Y = Y
        self.n_rollouts = self.Y.shape[0]
        self.indices = np.arange(self.n_rollouts)

    def get_batch(self, batch_size: int):
        batch_idxs = self.rng.choice(self.indices, size=batch_size, replace=False)
        return (self.U_enc[batch_idxs],
                self.U_dyn[batch_idxs],
                self.U_dec[batch_idxs],
                self.Y[batch_idxs])

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return (self.U_enc[idx], self.U_dyn[idx], self.U_dec[idx], self.Y[idx])


# ---------- Funciones auxiliares ----------
def load_vicon_marker_locations(path_to_yaml: str):
    with open(path_to_yaml, 'r') as file:
        marker_data = yaml.safe_load(file)

    marker_dist = {k: np.linalg.norm(v) for k, v in marker_data.items()}
    return marker_dist

def load_trajs(n_trajs: list):
    dataset_path = './datasets/dlo/'
    trajfile_names = [f'traj{n}.csv' for n in n_trajs]
    trajfile_paths = [dataset_path + t for t in trajfile_names]
    trajs = [DLOTrajectory(t, n) for t, n in zip(trajfile_paths, n_trajs)]
    return trajs

def load_trajs_old(n_trajs: list):
    dataset_path = './datasets/dlo_old/'
    trajfile_names = [f'traj{n}.csv' for n in n_trajs]
    trajfile_paths = [dataset_path + t for t in trajfile_names]
    trajs = [DLOTrajectory(t, n) for t, n in zip(trajfile_paths, n_trajs)]
    return trajs

def plot_trajs(T, Y, lbls: list):
    if not isinstance(T, list):
        T, Y = [T], [Y]

    n_cols = Y[0].shape[1] // 3
    _, axs = plt.subplots(3, n_cols, sharex=True)
    axs = axs.T.reshape(-1)

    for t, y in zip(T, Y):
        for k, ax in enumerate(axs):
            ax.plot(t, y[:, k])
            ax.set_ylabel(lbls[k])
            ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
