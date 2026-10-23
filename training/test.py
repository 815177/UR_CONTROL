import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.nets import SimpleEncoder, RnnDynamics, SimpleDecoder, DLOModel, LstmDynamics, GruDynamics, build_model_from_checkpoint
from training.process_data import *
from training.align import align_mocap_to_robot
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score


device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ LOAD MODEL -------------- #
checkpoint = torch.load("./models/model_simple.pth", map_location=device, weights_only=False)

model = build_model_from_checkpoint(checkpoint, device)
enc_history = checkpoint.get("enc_history", 1)      # nº de frames pasados que ve el encoder
model.eval()

Dyn_scaler = checkpoint["Dyn_scaler"]
Output_scaler = checkpoint["Output_scaler"]

vec = list(range(27, 35))
trajs = load_trajs(vec)
if checkpoint.get("align_mocap", False):
    trajs = align_mocap_to_robot(trajs)
horizon_test, _ = construct_train_val_datasets_from_trajs(
    trajs, rollout_length=160, val_size=0.5, seed=800, enc_history=enc_history
)

u_enc, U_dyn, U_dec, Y = horizon_test[450]
u_enc, U_dyn, U_dec, Y = (
    u_enc.to(device), U_dyn.to(device), U_dec.to(device), Y.to(device)
)

U_dyn = Dyn_scaler.transform(horizon_test.dyn_scalar.inverse_transform(U_dyn))

import time
start = time.time()
_, Y_pred = model(u_enc.unsqueeze(0), U_dyn.unsqueeze(0), U_dec.unsqueeze(0))
end = time.time()
print(end-start)

Y = Y.cpu()
Y_pred = Y_pred.cpu().detach()[0]

Y = horizon_test.output_scalar.inverse_transform(Y)
Y_pred = Output_scaler.inverse_transform(Y_pred)

mse_train = mean_squared_error(Y_pred[:, 0:3].reshape(-1, 3), Y[:, 0:3].reshape(-1, 3))
rmse_train = np.sqrt(mse_train)

print(rmse_train)

axis = ["X", "Y", "Z"]
for d in range(3):
    plt.figure(figsize=(8,4))
    plt.plot(Y[:, d].numpy().flatten(), label='True', alpha=0.7)
    plt.plot(Y_pred[:, d].numpy().flatten(), label='Pred', alpha=0.7)
    plt.title(f'8 seconds forecasting - ' + axis[d] + ' axis')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()