import numpy as np
import time 
import sys, os
import math
import torch

from core.trajectory import *
from core.optitrack import DLOTracker_Mean
from core.ur_control import * 
from core.conversions import *

from training.nets import *
from training.data import *
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------ LOAD MODEL -------------- #
checkpoint = torch.load("../models/model_simple.pth", map_location=device, weights_only=False)

model = build_model_from_checkpoint(checkpoint, device)
enc_history = checkpoint.get("enc_history", 1)      # nº de frames pasados que ve el encoder
enc_commands = checkpoint.get("enc_commands", False)  # el encoder ve también el último comando enviado

Dyn_scaler = checkpoint["Dyn_scaler"]
Output_scaler = checkpoint["Output_scaler"]

# Historia del encoder (si el modelo usa los últimos enc_history frames, uno por paso de control)
from collections import deque
enc_buffer = deque(maxlen=enc_history)
def encoder_input(u_enc, u_pitch_yaw):
    """Añade el frame actual a la historia y devuelve la entrada del encoder: (D,) o (enc_history, D)"""
    if enc_commands:
        u_enc = torch.cat([u_enc, u_pitch_yaw])
    if enc_history == 1:
        return u_enc
    if not enc_buffer:
        enc_buffer.extend([u_enc] * enc_history)
    else:
        enc_buffer.append(u_enc)
    return torch.stack(list(enc_buffer))

# -----------------------------------------------
ur10_control = UR10Control(host_ip='CMP00-180723AD.local')
time.sleep(2)

dloTracker = DLOTracker_Mean(client_address='169.254.126.181', server_address='169.254.126.180', device=device, num_markers=2)
dloTracker.start()

def mpc_step(u_enc, u_pitch_yaw, goal, initial_actions, gd_iters=30, lr=0.01, alpha=0.0):
    actions = initial_actions.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([actions], lr=lr)
  
    u_min = torch.tensor([-0.6, -0.6], device=actions.device)
    u_max = torch.tensor([ 0.6,  0.6], device=actions.device)

    # Solo el GRU en modo train (cuDNN lo exige para el backward); el dropout de encoder/decoder desactivado
    model.eval()
    model.dynamics.train()
    for p in model.parameters():
        p.requires_grad_(False)

    u_enc_input = u_enc.unsqueeze(0)
    goal_xy = goal[1:, :2]

    for _ in range(gd_iters):

        U_dec_input = torch.cat([u_pitch_yaw.unsqueeze(0), actions], dim=0)
        U_dyn_scaled = Dyn_scaler.transform(actions)

        # --- Preparar entradas ---
        U_dec_input = U_dec_input.unsqueeze(0)
        U_dyn_input = U_dyn_scaled.unsqueeze(0)

        # --- Forward del modelo ---
        _, pred = model(u_enc_input, U_dyn_input, U_dec_input)
        pred = pred.squeeze(0)
      
        # --- Coste ---
        err = goal_xy - pred[1:, :2]
        cost_ref = torch.sum(err * err)

        diff_u = actions[1:] - actions[:-1]
        cost_smooth = torch.sum(diff_u * diff_u)
        cost_smooth += torch.sum((u_pitch_yaw - actions[0])**2)

        penalty = torch.sum(F.relu(actions - u_max)**2 + F.relu(u_min - actions)**2)

        loss = cost_ref + alpha * cost_smooth + penalty * 100

        # --- Backward y optimización ---
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return actions.detach()


# Trajectory settings
dt = 0.05
n_secs = 20
H = 399
# ------------------------------------------

# Initial pose 
euler_init = [3.1323, 0.0487, 2.9804] 
q_rot = rpy2q(3.1323, 0.0487, 2.9804, 'rad') 
rot_init = q2r(q_rot)

T_init = np.eye(4) 
T_init[:3, 3] = [-0.17996171, 0.62444155, 0.49108581]
T_init[:3, :3] = q2r(q_rot)

# Setting initial rotation
time.sleep(1)

ur10_control.send_trajectory_smooth(T_init, dt=5)

time.sleep(10)

# Creating trajectory log
log_file = open(f"../datasets/control/traj{sys.argv[1]}.csv", "w")
header = (
        "t,"
        "p_b_x,p_b_y,p_b_z,"
        "phi_b_x,phi_b_y,phi_b_z,"
        "dp_b_x,dp_b_y,dp_b_z,"
        "dphi_b_x,dphi_b_y,dphi_b_z,"
        "p_e_x,p_e_y,p_e_z,"
        "dp_e_x,dp_e_y,dp_e_z,"
        "u_pitch,u_yaw\n"
    )
log_file.write(header)

# ================= State (positions) =================
left_f = dloTracker.marker_positions[1, :].clone()
right_f = dloTracker.marker_positions[0, :].clone()
orient_f = quat2euler(torch.tensor(ur10_control.q_current, device=device))

left_f_prev = left_f.clone()
right_f_prev = right_f.clone()
orient_f_prev = orient_f.clone()

# ================= Velocities =================
left_vel_f = torch.zeros(3, device=device)
right_vel_f = torch.zeros(3, device=device)
orient_vel_f = torch.zeros(3, device=device)

# ================= Filter params =================
alpha_p = 0.7      

# ================= Actions =================
u_pitch_yaw = torch.zeros(2, device=device, dtype=torch.float32)
last_control_input = torch.zeros(2, device=device, dtype=torch.float32)
actions = torch.zeros(H, 2, device=device, dtype=torch.float32)

# ================= GENERATE DESIRED TRAJECTORY =================
t, x, y = generar_trayectoria('circulo', velocidad=1.25, radio=0.25, centro=(1, 0.9), dt=dt)   #209
# t, x, y = generar_trayectoria('corazon', velocidad=0.5, escala=0.20, centro=(1, 0.9), dt=dt)  #210
# t, x, y = generar_trayectoria('cuadrado', velocidad=0.8, escala=0.20, centro=(1.0,0.9), dt=dt) #211 212
# t, x, y = generar_trayectoria('infinito', velocidad=0.8, escala=0.2, centro=(1, 0.9), dt=dt)   #213
# t, x, y = generar_trayectoria('linea_y', velocidad=1, escala=0.4, centro=(1, 0.9), dt=dt)
# t, x, y = generar_trayectoria('linea_z', velocidad=1, escala=0.3, centro=(1, 0.9), dt=dt)
# t, x, y = generar_trayectoria('diagonal', velocidad=1, escala=0.35, centro=(1, 0.9), dt=dt)

zeros = np.zeros(len(t))
dlo_path = np.column_stack((x, y, zeros))
dlo_path = torch.tensor(dlo_path, device=device, dtype=torch.float32)
 
# ================= INITIAL MPC STEP =================
orient_sin_cos = compute_sin_cos_gpu(orient_f)
ang_vel_sin_cos = torch.cat([
    orient_sin_cos[3:] * orient_vel_f,
    -orient_sin_cos[:3] * orient_vel_f
], dim=-1)

u_enc = torch.cat([
    right_f - left_f,
    orient_sin_cos,
    right_vel_f - left_vel_f,
    ang_vel_sin_cos
], dim=-1)

end_idx = H + 1
Y = Output_scaler.transform(dlo_path[:end_idx])

actions = mpc_step(encoder_input(u_enc, u_pitch_yaw), u_pitch_yaw, Y, actions,  gd_iters=5000, lr=95e-4, alpha=1.5)

# ================ PLOTTING INITIAL ACTIONS =================
actions_clone = actions.clone()
actions_plot = actions_clone.cpu().numpy()
plt.figure(figsize=(8,4))
plt.plot(actions_plot[:,0], label='Action 1')
plt.plot(actions_plot[:,1], label='Action 2')
plt.title('Optimized Actions over Horizon')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Plot predicted vs goal
U_dyn_scaled = Dyn_scaler.transform(actions)
U_dyn_input = U_dyn_scaled.unsqueeze(0)
U_dec_input = torch.cat([u_pitch_yaw.unsqueeze(0), actions], dim=0).unsqueeze(0)
u_enc_input = (torch.stack(list(enc_buffer)) if enc_history > 1 else
               (torch.cat([u_enc, u_pitch_yaw]) if enc_commands else u_enc)).unsqueeze(0)

with torch.no_grad():
    _, pred = model(u_enc_input, U_dyn_input.to(device), U_dec_input.to(device))
pred =  Output_scaler.inverse_transform(pred.squeeze(0).cpu().numpy())
goal_xy =  Output_scaler.inverse_transform(Y[1:].cpu().numpy())
plt.figure(figsize=(8,8))
plt.plot(range(H), goal_xy[:,0], label='Goal Path X',)
plt.plot(range(H), goal_xy[:,1], label='Goal Path Y')
plt.plot(range(H+1), pred[:,0], label='Predicted Path X')
plt.plot(range(H+1), pred[:,1], label='Predicted Path Y')
plt.title('MPC: Predicted vs Goal Path')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

time.sleep(10)
# =================================================================

# ================= MAIN LOOP =================
t_prev = time.perf_counter()
next_tick = t_prev  + dt
time_init = time.time()
time.sleep(dt)
for i in range(H - 1):

    t = time.time() - time_init

    # ================= TIME =================
    t_now = time.perf_counter()
    dt_real = t_now - t_prev
    t_prev = t_now

    # ================= MEASUREMENTS =================
    left_new = dloTracker.marker_positions[1, :].clone()
    right_new = dloTracker.marker_positions[0, :].clone()
    orient_new = quat2euler(torch.tensor(ur10_control.q_current, device=device))
    
    # unwrap angular difference
    dtheta = orient_new[0] - orient_f[0]
    dtheta = torch.remainder(dtheta + math.pi, 2 * math.pi) - math.pi
    orient_new[0] = orient_f[0] + dtheta

    # ================= FILTER POSITIONS =================
    left_f = alpha_p * left_new + (1 - alpha_p) * left_f
    right_f = alpha_p * right_new + (1 - alpha_p) * right_f
    orient_f = alpha_p * orient_new + (1 - alpha_p) * orient_f

    # ================= VELOCITIES =================
    left_vel_new = (left_f - left_f_prev) / dt_real
    right_vel_new = (right_f - right_f_prev) / dt_real
    orient_vel_new = (orient_f - orient_f_prev) / dt_real

    # ================= Updating State =================    
    left_vel_f = left_vel_new.clone()
    right_vel_f = right_vel_new.clone()
    orient_vel_f = orient_vel_new.clone()
    u_pitch_yaw = last_control_input.clone()
    left_f_prev = left_f.clone()
    right_f_prev = right_f.clone()
    orient_f_prev = orient_f.clone()
    
    # ================= LOGGING =================
    line = (
        f"{t:.6f},"
        f"{left_f[0].item():.10f},{left_f[1].item():.10f},{left_f[2].item():.10f},"
        f"{orient_f[0].item():.10f},{orient_f[1].item():.10f},{orient_f[2].item():.10f},"
        f"{left_vel_f[0].item():.6f},{left_vel_f[1].item():.6f},{left_vel_f[2].item():.6f},"
        f"{orient_vel_f[0].item():.6f},{orient_vel_f[1].item():.6f},{orient_vel_f[2].item():.6f},"
        f"{right_f[0].item():.6f},{right_f[1].item():.6f},{right_f[2].item():.6f},"
        f"{right_vel_f[0].item():.6f},{right_vel_f[1].item():.6f},{right_vel_f[2].item():.6f},"
        f"{u_pitch_yaw[0].item():.10f},{u_pitch_yaw[1].item():.10f}\n"
    )
    
    log_file.write(line)
    log_file.flush()

    # ================= SLEPP?? =================    
    time.sleep(0.025)

    # ================= UPDATE CONTROL INPUT =================
    last_control_input = actions[i].clone()

    # ================= SEND TRAJECTORY =================
    a = last_control_input.cpu().numpy()
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy2r(euler_init[0] + a[0], euler_init[1], euler_init[2] + a[1])
    T[:3, 3] = T_init[:3, 3]
    ur10_control.send_trajectory_smooth(T, dt= dt)
   
    # ================= TIMING =================
    next_tick += dt
    sleep_time = next_tick - time.perf_counter()
    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        next_tick = time.perf_counter()

log_file.close()
os._exit(0)


