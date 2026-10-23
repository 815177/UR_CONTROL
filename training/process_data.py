# process_data_torch.py
import numpy as np
import torch
from typing import List
import itertools
from training.data import *

# ---------- helpers ----------
def compute_sin_cos_features(phi: np.ndarray):
    return np.hstack([np.sin(phi), np.cos(phi)])


# ---------- encoder ----------
def construct_encoder_inputs(trajs: List[DLOTrajectory]):
    feats = []
    for traj in trajs:
        p_be = traj.p_e - traj.p_b                  
        orient = compute_sin_cos_features(traj.phi_b)
        lin_vel = traj.dp_e - traj.dp_b            
        ang_vel = np.hstack([
            orient[:, 3:] * traj.dphi_b,
            -orient[:, :3] * traj.dphi_b
        ])
        feats.append(np.hstack([p_be, orient, lin_vel, ang_vel]))
    return feats


def construct_encoder_inputs2(trajs: List[DLOTrajectory]):
    feats = []
    for traj in trajs:
        orient = compute_sin_cos_features(traj.phi_b)
        ang_vel = np.hstack([
            orient[:, 3:] * traj.dphi_b,
            -orient[:, :3] * traj.dphi_b
        ])
        feats.append(np.hstack([traj.p_b, traj.p_e, orient, traj.dp_b, traj.dp_e, ang_vel]))
    return feats

# ---------- dynamics ----------
def construct_dyn_inputs(trajs: List[DLOTrajectory]):
    feats = []
    for traj in trajs:
        feats.append(np.hstack([traj.u_pitch_yaw]))
    return feats


# ---------- decoder ----------
def construct_decoder_inputs(trajs: List[DLOTrajectory]):
    feats = []
    for traj in trajs:
       feats.append(np.hstack([traj.u_pitch_yaw]))
    return feats

# ---------- outputs ----------
def construct_observations(trajs: List[DLOTrajectory]):
    return [np.hstack([traj.p_e]) for traj in trajs]

def construct_observations_force(trajs: List[DLOTrajectory]):
    return [np.hstack([traj.p_e, traj.force]) for traj in trajs]

# ---------- ventana ----------
def rolling_window(traj_length: int, window: int):
    n_windows = traj_length - window + 1
    return np.arange(n_windows)[:, None] + np.arange(window)[None, :]

def divide_into_rolling_windows(trajs, window: int):
    rollouts = []
    for traj in trajs:
        if traj.shape[0] > window:
            idxs = rolling_window(traj.shape[0], window + 1)
            rollouts.append(traj[idxs])
    return rollouts


def divide_into_rolling_windows_and_stack(trajs, window: int):
    rollouts = divide_into_rolling_windows(trajs, window)
    return torch.from_numpy(np.vstack(rollouts)).float()


# ---------- slicing ----------
def slice_trajs(trajs, idxs):
    s_trajs = [[traj.get_sliced_copy(sl) for sl in idxs_k]
               for traj, idxs_k in zip(trajs, idxs)]
    return list(itertools.chain.from_iterable(s_trajs))


def train_val_split_idxs(len_traj: int, ratio: float, rng: np.random.Generator):
    
    val_len = int(len_traj * ratio)
    train_len = len_traj - val_len

    val_idxs = [np.arange(train_len, len_traj)]
    train_idxs = [np.arange(0, train_len)]

    return train_idxs, val_idxs


def get_train_val_idxs(trajs, val_size: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idxs, val_idxs = [], []
    
    for traj in trajs:
        tr_idx, val_idx = train_val_split_idxs(len(traj), ratio=val_size, rng=rng)
        train_idxs.append(tr_idx)
        val_idxs.append(val_idx)
        
    return train_idxs, val_idxs

# ---------- construcción dataset ----------
def construct_train_val_datasets_from_trajs(trajs, rollout_length: int, val_size: float, seed: int):
    # split
    train_idxs, val_idxs = get_train_val_idxs(trajs, val_size, seed)
    train_trajs = slice_trajs(trajs, train_idxs)
    val_trajs = slice_trajs(trajs, val_idxs)

    # encoder (sin escalar: DLOModel aplica enc_scalar en el forward)
    enc_train = construct_encoder_inputs(train_trajs)
    enc_val = construct_encoder_inputs(val_trajs)
    enc_scalar = TorchStandardScaler()
    enc_scalar.fit(np.vstack(enc_train))
    U_enc_train = divide_into_rolling_windows_and_stack(enc_train, rollout_length)
    U_enc_val = divide_into_rolling_windows_and_stack(enc_val, rollout_length)

    # dynamics
    dyn_train = construct_dyn_inputs(train_trajs)
    dyn_val = construct_dyn_inputs(val_trajs)
    dyn_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    dyn_scalar.fit(np.vstack(dyn_train))
    dyn_train_scaled = [dyn_scalar.transform(t) for t in dyn_train]
    dyn_val_scaled = [dyn_scalar.transform(t) for t in dyn_val]
    U_dyn_train = divide_into_rolling_windows_and_stack(dyn_train_scaled, rollout_length)[:, 1:, :]
    U_dyn_val = divide_into_rolling_windows_and_stack(dyn_val_scaled, rollout_length)[:, 1:, :]

    # decoder
    dec_train = construct_decoder_inputs(train_trajs)
    dec_val = construct_decoder_inputs(val_trajs)
    dec_scalar = TorchMinMaxScaler(feature_range=(-1, 1))   # sin escalar: DLOModel aplica dec_scalar
    dec_scalar.fit(np.vstack(dec_train))
    U_dec_train = divide_into_rolling_windows_and_stack(dec_train, rollout_length)
    U_dec_val = divide_into_rolling_windows_and_stack(dec_val, rollout_length)

    # outputs
    obs_train = construct_observations(train_trajs)
    obs_val = construct_observations(val_trajs)
    obs_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    obs_scalar.fit(np.vstack(obs_train))
    obs_train_scaled = [obs_scalar.transform(t) for t in obs_train]
    obs_val_scaled = [obs_scalar.transform(t) for t in obs_val]
    Y_train = divide_into_rolling_windows_and_stack(obs_train_scaled, rollout_length)
    Y_val = divide_into_rolling_windows_and_stack(obs_val_scaled, rollout_length)

    # dataset
    train_ds = TorchDLODataset(U_encoder=U_enc_train,
                          U_dyn=U_dyn_train,
                          U_decoder=U_dec_train,
                          Y=Y_train,
                          encoder_scalar=enc_scalar,
                          dyn_scalar=dyn_scalar,
                          output_scalar=obs_scalar,
                          decoder_scalar=dec_scalar)

    val_ds = TorchDLODataset(U_encoder=U_enc_val,
                        U_dyn=U_dyn_val,
                        U_decoder=U_dec_val,
                        Y=Y_val,
                        encoder_scalar=enc_scalar,
                        dyn_scalar=dyn_scalar,
                        output_scalar=obs_scalar,
                        decoder_scalar=dec_scalar)

    return train_ds, val_ds

def construct_train_val_datasets_from_trajs_ft(trajs, rollout_length: int, val_size: float, seed: int):
    # split
    train_idxs, val_idxs = get_train_val_idxs(trajs, val_size, seed)
    train_trajs = slice_trajs(trajs, train_idxs)
    val_trajs = slice_trajs(trajs, val_idxs)

    # encoder -> Normalizar tiambién????
    enc_train = construct_encoder_inputs(train_trajs)
    enc_val = construct_encoder_inputs(val_trajs)
    U_enc_train = divide_into_rolling_windows_and_stack(enc_train, rollout_length)
    U_enc_val = divide_into_rolling_windows_and_stack(enc_val, rollout_length)

    # dynamics
    dyn_train = construct_dyn_inputs(train_trajs)
    dyn_val = construct_dyn_inputs(val_trajs)
    dyn_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    dyn_scalar.fit(np.vstack(dyn_train))
    dyn_train_scaled = [dyn_scalar.transform(t) for t in dyn_train]
    dyn_val_scaled = [dyn_scalar.transform(t) for t in dyn_val]
    U_dyn_train = divide_into_rolling_windows_and_stack(dyn_train_scaled, rollout_length)[:, 1:, :]
    U_dyn_val = divide_into_rolling_windows_and_stack(dyn_val_scaled, rollout_length)[:, 1:, :]

    # decoder
    dec_train = construct_decoder_inputs(train_trajs)
    dec_val = construct_decoder_inputs(val_trajs)
    U_dec_train = divide_into_rolling_windows_and_stack(dec_train, rollout_length)
    U_dec_val = divide_into_rolling_windows_and_stack(dec_val, rollout_length)

    # outputs
    obs_train = construct_observations_force(train_trajs)
    obs_val = construct_observations_force(val_trajs)
    obs_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    obs_scalar.fit(np.vstack(obs_train))
    obs_train_scaled = [obs_scalar.transform(t) for t in obs_train]
    obs_val_scaled = [obs_scalar.transform(t) for t in obs_val]
    Y_train = divide_into_rolling_windows_and_stack(obs_train_scaled, rollout_length)
    Y_val = divide_into_rolling_windows_and_stack(obs_val_scaled, rollout_length)

    # dataset
    train_ds = TorchDLODataset(U_encoder=U_enc_train,
                          U_dyn=U_dyn_train,
                          U_decoder=U_dec_train,
                          Y=Y_train,
                          encoder_scalar=None,
                          dyn_scalar=dyn_scalar,
                          output_scalar=obs_scalar)

    val_ds = TorchDLODataset(U_encoder=U_enc_val,
                        U_dyn=U_dyn_val,
                        U_decoder=U_dec_val,
                        Y=Y_val,
                        encoder_scalar=None,
                        dyn_scalar=dyn_scalar,
                        output_scalar=obs_scalar)

    return train_ds, val_ds


def construct_train_val_datasets_from_trajs2(trajs, rollout_length: int, val_size: float, seed: int):
    # split
    train_idxs, val_idxs = get_train_val_idxs(trajs, val_size, seed)
    train_trajs = slice_trajs(trajs, train_idxs)
    val_trajs = slice_trajs(trajs, val_idxs)

    # encoder 
    enc_train = construct_encoder_inputs2(train_trajs)
    enc_val = construct_encoder_inputs2(val_trajs)
    U_enc_train = divide_into_rolling_windows_and_stack(enc_train, rollout_length)
    U_enc_val = divide_into_rolling_windows_and_stack(enc_val, rollout_length)

    # dynamics
    dyn_train = construct_dyn_inputs(train_trajs)
    dyn_val = construct_dyn_inputs(val_trajs)
    dyn_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    dyn_scalar.fit(np.vstack(dyn_train))
    dyn_train_scaled = [dyn_scalar.transform(t) for t in dyn_train]
    dyn_val_scaled = [dyn_scalar.transform(t) for t in dyn_val]
    U_dyn_train = divide_into_rolling_windows_and_stack(dyn_train_scaled, rollout_length)[:, 1:, :]
    U_dyn_val = divide_into_rolling_windows_and_stack(dyn_val_scaled, rollout_length)[:, 1:, :]

    # decoder
    dec_train = construct_decoder_inputs(train_trajs)
    dec_val = construct_decoder_inputs(val_trajs)
    U_dec_train = divide_into_rolling_windows_and_stack(dec_train, rollout_length)
    U_dec_val = divide_into_rolling_windows_and_stack(dec_val, rollout_length)

    # outputs
    obs_train = construct_observations(train_trajs)
    obs_val = construct_observations(val_trajs)
    obs_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    obs_scalar.fit(np.vstack(obs_train))
    obs_train_scaled = [obs_scalar.transform(t) for t in obs_train]
    obs_val_scaled = [obs_scalar.transform(t) for t in obs_val]
    Y_train = divide_into_rolling_windows_and_stack(obs_train_scaled, rollout_length)
    Y_val = divide_into_rolling_windows_and_stack(obs_val_scaled, rollout_length)

    # dataset
    train_ds = TorchDLODataset(U_encoder=U_enc_train,
                          U_dyn=U_dyn_train,
                          U_decoder=U_dec_train,
                          Y=Y_train,
                          encoder_scalar=None,
                          dyn_scalar=dyn_scalar,
                          output_scalar=obs_scalar)

    val_ds = TorchDLODataset(U_encoder=U_enc_val,
                        U_dyn=U_dyn_val,
                        U_decoder=U_dec_val,
                        Y=Y_val,
                        encoder_scalar=None,
                        dyn_scalar=dyn_scalar,
                        output_scalar=obs_scalar)

    return train_ds, val_ds


def construct_train_val_datasets_from_trajs_(trajs, rollout_length: int, val_size: float, seed: int):
    # split
    train_idxs, val_idxs = get_train_val_idxs(trajs, 0.0, seed)
    train_trajs = slice_trajs(trajs, train_idxs)

    # encoder -> Normalizar tiambién????
    enc_train = construct_encoder_inputs(train_trajs)
    U_enc_train = divide_into_rolling_windows_and_stack(enc_train, rollout_length)

    # dynamics
    dyn_train = construct_dyn_inputs(train_trajs)
    dyn_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    dyn_scalar.fit(np.vstack(dyn_train))
    dyn_train_scaled = [dyn_scalar.transform(t) for t in dyn_train]
    U_dyn_train = divide_into_rolling_windows_and_stack(dyn_train_scaled, rollout_length)[:, 1:, :]

    # decoder
    dec_train = construct_decoder_inputs(train_trajs)
    U_dec_train = divide_into_rolling_windows_and_stack(dec_train, rollout_length)

    # outputs
    obs_train = construct_observations(train_trajs)
    obs_scalar = TorchMinMaxScaler(feature_range=(-1, 1))
    obs_scalar.fit(np.vstack(obs_train))
    obs_train_scaled = [obs_scalar.transform(t) for t in obs_train]
    Y_train = divide_into_rolling_windows_and_stack(obs_train_scaled, rollout_length)

    # dataset
    train_ds = TorchDLODataset(U_encoder=U_enc_train,
                          U_dyn=U_dyn_train,
                          U_decoder=U_dec_train,
                          Y=Y_train,
                          encoder_scalar=None,
                          dyn_scalar=dyn_scalar,
                          output_scalar=obs_scalar)

    return train_ds