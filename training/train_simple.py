import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.nets import SimpleEncoder, RnnDynamics, SimpleDecoder, DLOModel, LstmDynamics, GruDynamics, DLOModel
from training.process_data import *
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

class RolloutScheduler:
    def __init__(self, init_value=0.1, boundaries=None, scales=None, max_rollout=None):
   
        self.init_value = init_value
        self.boundaries = boundaries or []
        self.scales = scales or []
        self.max_rollout = max_rollout

    def get_rollout_horizon(self, epoch):
        val = self.init_value
        for b, s in zip(self.boundaries, self.scales):
            if epoch >= b:
                val *= s
        if self.max_rollout is not None:
            val = min(val, self.max_rollout)
        return int(val)
    
def evaluate_and_plot(model, train_ds, val_ds, batch_size=16, device='cuda', save_dir="plots"):
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True) 
    
    def forward_dataset(ds):
        Y_preds = []
        n_rollouts = ds.U_encoder.shape[0]
        for i in range(0, n_rollouts, batch_size):
            u_enc_batch = ds.U_encoder[i:i+batch_size].to(device)
            U_dyn_batch = ds.U_dyn[i:i+batch_size].to(device)
            U_dec_batch = ds.U_decoder[i:i+batch_size].to(device)
            with torch.no_grad():
                _, Y_pred = model(u_enc_batch[:, 0, :], U_dyn_batch, U_dec_batch)
            Y_preds.append(Y_pred.cpu())
        return torch.cat(Y_preds, dim=0)

    Y_train_pred = forward_dataset(train_ds)
    Y_val_pred = forward_dataset(val_ds)
    
    Y_train_true = train_ds.Y
    Y_val_true = val_ds.Y

    print(f"Train set: True Y shape: {Y_train_true.shape}, Pred Y shape: {Y_train_pred.shape}")
    print(f"Val set: True Y shape: {Y_val_true.shape}, Pred Y shape: {Y_val_pred.shape}")

    n_dims = 3

    # --- Plot Train ---
    train_dir = os.path.join(save_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    for d in range(n_dims):
        plt.figure(figsize=(8,4))
        plt.plot(Y_train_true[:, 5, d].numpy().flatten(), label='True', alpha=0.7)
        plt.plot(Y_train_pred[:, 5, d].numpy().flatten(), label='Pred', alpha=0.7)
        plt.title(f'Train - Dimension {d}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, f"dim_{d}.png"), dpi=200)
        plt.close()

    # --- Plot Val ---
    val_dir = os.path.join(save_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for d in range(n_dims):
        plt.figure(figsize=(8,4))
        plt.plot(Y_val_true[:, 5, d].numpy().flatten(), label='True', alpha=0.7)
        plt.plot(Y_val_pred[:, 5, d].numpy().flatten(), label='Pred', alpha=0.7)
        plt.title(f'Val - Dimension {d}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(val_dir, f"dim_{d}.png"), dpi=200)
        plt.close()

    # --- Métricas ---
    Y_train_pred = train_ds.output_scalar.inverse_transform(Y_train_pred)
    Y_train_true = train_ds.output_scalar.inverse_transform(Y_train_true)
    Y_val_pred = train_ds.output_scalar.inverse_transform(Y_val_pred)
    Y_val_true = train_ds.output_scalar.inverse_transform(Y_val_true)
    
    mse_train = mean_squared_error(Y_train_pred[:, :, 0:3].reshape(-1, 3), Y_train_true[:, :, 0:3].reshape(-1, 3))
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(Y_train_pred.reshape(-1, 3), Y_train_true.reshape(-1, 3))

    print("MSE TRAIN: ", mse_train)
    print("RMSE TRAIN: ", rmse_train)
    print("R² TRAIN: ", r2_train)

    mse_val = mean_squared_error(Y_val_pred[:, :, 0:3].reshape(-1, 3), Y_val_true[:, :, 0:3].reshape(-1, 3))
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(Y_val_pred.reshape(-1, 3), Y_val_true.reshape(-1, 3))

    print("MSE VAL: ", mse_val)
    print("RMSE VAL: ", rmse_val)
    print("R² VAL: ", r2_val)

    return Y_train_pred, Y_val_pred

def train_model(model, train_ds, val_ds, n_epochs=25, batch_size=64, device="cuda"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device) 

    optimizer = optim.Adam(model.parameters(),  lr=0.0005)
    criterion = nn.L1Loss(reduction='none')              # nn.MSELoss() nn.L1Loss()

    # rollout scheduler como en el paper
    scheduler_cfg = {
        "init_value": 10,
        "boundaries": [30, 45, 60, 75],
        "scales": [1.5, 1.5, 1.5, 1.5]
    }

    # scheduler_cfg = {
    #     "init_value": 80,
    #     "boundaries": [40, 80, 120, 160],
    #     "scales": [1, 1, 1, 1]
    # }
    scheduler = RolloutScheduler(**scheduler_cfg, max_rollout=train_ds.Y.shape[1])

    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0.0

        rollout_horizon = scheduler.get_rollout_horizon(epoch)

        for u_enc, U_dyn, U_dec, Y in train_loader:
            u_enc, U_dyn, U_dec, Y = (
                u_enc.to(device), U_dyn.to(device), U_dec.to(device), Y.to(device)
            )

            optimizer.zero_grad()
            _, Y_pred = model(u_enc[:, 0, :], U_dyn, U_dec)

            # Aplicar solo hasta rollout_horizon
            # loss = criterion(Y_pred[:, :rollout_horizon], Y[:, :rollout_horizon])
            loss_per_seq = torch.mean(torch.abs(Y_pred[:, :rollout_horizon, :] - Y[:, :rollout_horizon, :]), dim=(1,2)) 
            loss = loss_per_seq.mean()
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * u_enc.size(0)

        train_loss /= len(train_loader.dataset)

        # validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u_enc, U_dyn, U_dec, Y in val_loader:
                u_enc, U_dyn, U_dec, Y = (
                    u_enc.to(device), U_dyn.to(device), U_dec.to(device), Y.to(device)
                )
                _, Y_pred = model(u_enc[:, 0, :], U_dyn, U_dec)
                # val_loss += criterion(
                #     Y_pred[:, :rollout_horizon], Y[:, :rollout_horizon]
                # ).item() * u_enc.size(0)
                val_loss += ((torch.mean(torch.abs(Y_pred[:, :rollout_horizon, :] - Y[:, :rollout_horizon, :]), dim=(1,2))).mean()).item() * u_enc.size(0) 

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:03d} | rollout={rollout_horizon} | "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return model


device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar trayectorias
vec = list(range(24)) + list(range(37, 41)) + list(range(44, 50))
# vec = list(range(150))
# trajs = load_trajs_old(vec)
trajs = load_trajs(vec)

# Construir modelo -> corregir dimensiones según datos
encoder = SimpleEncoder(input_dim=18, hidden_dims=[32, 32], output_dim=160, device=device)
dynamics = GruDynamics(input_dim=2, hidden_dim=160, n_layers=1, device=device)
decoder = SimpleDecoder(input_dim=162, hidden_dims=[32, 32], output_dim=3, device=device)
model = DLOModel(encoder, dynamics, decoder, n_layers=1)

torch.cuda.empty_cache()

train_ds, val_ds = construct_train_val_datasets_from_trajs(
    trajs, rollout_length=40, val_size=0.2, seed=800
)

# Entrenamiento
trained_model = train_model(model, train_ds, val_ds, n_epochs=90, batch_size=64, device=device)

# Evaluación
Y_train_pred, Y_val_pred = evaluate_and_plot(trained_model, train_ds, val_ds, batch_size=64, device=device)

checkpoint = {
    "model_state": model.state_dict(), 
    "model_params": {
        "encoder_params": {
            "input_dim": 18,
            "hidden_dims": [64, 64],
            "output_dim": 160
        },
        "dynamics_params": {
            "input_dim": 2,
            "hidden_dim":160,
            "n_layers": 1
        },
        "decoder_params": {
            "input_dim": 162,
            "hidden_dims": [64, 64],
            "output_dim": 3
        }
    },
    "Dyn_scaler": train_ds.dyn_scalar,
    "Output_scaler": train_ds.output_scalar
}

torch.save(checkpoint, "./models/model_simple.pth")

horizon_test, _ = construct_train_val_datasets_from_trajs(
    trajs, rollout_length=160, val_size=0.5, seed=800
)

u_enc, U_dyn, U_dec, Y = horizon_test[150]
u_enc, U_dyn, U_dec, Y = (
    u_enc.to(device), U_dyn.to(device), U_dec.to(device), Y.to(device)
)

U_dyn = train_ds.dyn_scalar.transform(horizon_test.dyn_scalar.inverse_transform(U_dyn))

import time
start = time.time()
_, Y_pred = model(u_enc[0].unsqueeze(0), U_dyn.unsqueeze(0), U_dec.unsqueeze(0))
end = time.time()
print(end-start)

Y = Y.cpu()
Y_pred = Y_pred.cpu().detach()[0]

Y = horizon_test.output_scalar.inverse_transform(Y)
Y_pred = train_ds.output_scalar.inverse_transform(Y_pred)

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


Y_preds = []
n_rollouts = val_ds.U_encoder.shape[0]
for i in range(0, n_rollouts, 16):
    u_enc_batch = val_ds.U_encoder[i:i+16].to(device)
    U_dyn_batch = val_ds.U_dyn[i:i+16].to(device)
    U_dec_batch = val_ds.U_decoder[i:i+16].to(device)
    with torch.no_grad():
        _, Y_pred = model(u_enc_batch[:, 0, :], U_dyn_batch, U_dec_batch)
    Y_preds.append(Y_pred.cpu())
torch.cat(Y_preds, dim=0)

Y = val_ds.Y
Y = val_ds.output_scalar.inverse_transform(Y)
Y_pred = torch.cat(Y_preds, dim=0)
Y_pred = train_ds.output_scalar.inverse_transform(Y_pred)


for d in range(3):
    plt.figure(figsize=(8,4))
    plt.plot(Y[:, 5, d].numpy().flatten(), label='True', alpha=0.7)
    plt.plot(Y_pred[:, 5, d].numpy().flatten(), label='Pred', alpha=0.7)
    plt.title(f'Val - Dimension {d}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
plt.show()

errors = torch.mean(torch.abs(Y - Y_pred), dim=(1,2))

k = min(5, Y.shape[0])
topk_idx = torch.topk(errors, k).indices

plt.figure(figsize=(15, 3*k))

for i, idx in enumerate(topk_idx):
    plt.subplot(k, 1, i+1)
    seq_len, dims = Y.shape[1], Y.shape[2]
    for d in range(dims):
        plt.plot(Y[idx, :, d].numpy(), label=f'GT dim {d}', linestyle='-')
        plt.plot(Y_pred[idx, :, d].numpy(), label=f'Pred dim {d}', linestyle='--')
    plt.title(f'Sample {idx} - Error medio: {errors[idx]:.4f}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


lowest_idx = torch.topk(errors, k, largest=False).indices
plt.figure(figsize=(15, 3*k))
for i, idx in enumerate(lowest_idx):
    plt.subplot(k, 1, i+1)
    seq_len, dims = Y.shape[1], Y.shape[2]
    for d in range(dims):
        plt.plot(Y[idx, :, d].numpy(), label=f'GT dim {d}', linestyle='-')
        plt.plot(Y_pred[idx, :, d].numpy(), label=f'Pred dim {d}', linestyle='--')
    plt.title(f'Sample {idx} - Error medio: {errors[idx]:.4f}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()