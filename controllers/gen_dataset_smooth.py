import numpy as np
import time 
import sys, os
import math
import matplotlib.pyplot as plt
import torch

from core.trajectory import *
from core.optitrack import DLOTracker_Mean
from core.ur_control import * 
from core.conversions import *

device = "cuda" if torch.cuda.is_available() else "cpu"

ur10_control = UR10Control(host_ip='192.168.0.200')
time.sleep(2)

dloTracker = DLOTracker_Mean(client_address='169.254.126.181', server_address='169.254.126.180', device=device, num_markers=2)
dloTracker.start()

# Trajectory settings
dt = 0.05
n_secs = 30
ramp_time = 1.5
total_steps = int(n_secs / dt) + int(ramp_time / dt)

# Initial pose  -> COMPROBAR PORQUE EL ROBOT HA CAMBIADO DE POSE

euler_init = [3.1323, 0.0487, 2.9804] 
q_rot = rpy2q(3.1323, 0.0487, 2.9804, 'rad') 
rot_init = q2r(q_rot)

T_init = np.eye(4) 
T_init[:3, 3] = [-0.17996171, 0.62444155, 0.49108581]
T_init[:3, :3] = q2r(q_rot)

for traj_id in range(int(sys.argv[1]), int(sys.argv[2])): 

    # Setting initial pose
    time.sleep(1)

    ur10_control.send_trajectory_smooth(T_init, dt=7)

    time.sleep(10)

    # Creating trajectory
    t, traj = sample_angular_trajectory(duration=n_secs, dt=dt, random=True, ramp_time=ramp_time)

    # Plot trajectory
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.rad2deg(traj["pitch"]), label="Pitch (°)")
    plt.plot(t, np.rad2deg(traj["yaw"]), label="Yaw (°)")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [°]")
    plt.title("Pitch and Yaw Trajectories")
    plt.legend()
    plt.grid()
    plt.savefig(f"./datasets/plot/traj{traj_id}.png")
    plt.close()

    orient_traj = torch.zeros((total_steps, 2), device=device, dtype=torch.float32)
    for step in range(total_steps):

        pitch = traj["pitch"][step]
        yaw = traj["yaw"][step]
        
        euler_new = euler_init.copy()
        euler_new[0] += pitch 
        euler_new[2] += yaw 

        orient_traj[step, 0] = pitch
        orient_traj[step, 1] = yaw
        
    # Creating trajectory log
    log_file = open(f"./datasets/dlo/traj{traj_id}.csv", "w")
    header = (
            "t,"
            "p_b_x,p_b_y,p_b_z,"
            "phi_b_x,phi_b_y,phi_b_z,"
            "dp_b_x,dp_b_y,dp_b_z,"
            "dphi_b_x,dphi_b_y,dphi_b_z,"
            "p_e_x,p_e_y,p_e_z,"
            "dp_e_x,dp_e_y,dp_e_z,"
            "u_pitch,u_yaw,"
            "mocap_frame,mocap_age,mocap_valid_e,mocap_valid_b,robot_age\n"
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

    # ================= MAIN LOOP =================
    t_prev = time.perf_counter()
    next_tick = t_prev  + dt
    time_init = time.time()
    time.sleep(dt)
    for step in range(total_steps):

        t = time.time() - time_init

        # ================= TIME =================
        t_now = time.perf_counter()
        dt_real = t_now - t_prev
        t_prev = t_now

        # ================= MEASUREMENTS =================
        left_new = dloTracker.marker_positions[1, :].clone()
        right_new = dloTracker.marker_positions[0, :].clone()
        orient_new = quat2euler(torch.tensor(ur10_control.q_current, device=device))
        mocap_state = dloTracker.get_state()                          # edad del dato -> retardo medible offline
        robot_age = time.perf_counter() - ur10_control.joint_states_t
       
        # unwrap angular difference
        dtheta = orient_new[[0, 2]] - orient_f[[0 ,2]]
        dtheta = torch.remainder(dtheta + math.pi, 2 * math.pi) - math.pi
        orient_new[[0, 2]] = orient_f[[0, 2]] + dtheta
     
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

        # ================= Logging ================= 
        line = (
            f"{t:.6f},"
            f"{left_f[0].item():.10f},{left_f[1].item():.10f},{left_f[2].item():.10f},"
            f"{orient_f[0].item():.10f},{orient_f[1].item():.10f},{orient_f[2].item():.10f},"
            f"{left_vel_f[0].item():.6f},{left_vel_f[1].item():.6f},{left_vel_f[2].item():.6f},"
            f"{orient_vel_f[0].item():.6f},{orient_vel_f[1].item():.6f},{orient_vel_f[2].item():.6f},"
            f"{right_f[0].item():.6f},{right_f[1].item():.6f},{right_f[2].item():.6f},"
            f"{right_vel_f[0].item():.6f},{right_vel_f[1].item():.6f},{right_vel_f[2].item():.6f},"
            f"{u_pitch_yaw[0].item():.10f},{u_pitch_yaw[1].item():.10f},"
            f"{mocap_state['frame_number']},{mocap_state['age']:.4f},"
            f"{int(mocap_state['valid'][0])},{int(mocap_state['valid'][1])},{robot_age:.4f}\n"
        )

        log_file.write(line)
        log_file.flush()

        # ================= SLEPP?? =================
        time.sleep(0.025)

        # ================= UPDATE CONTROL INPUT =================
        last_control_input = orient_traj[step].clone()

        # ================= SEND TRAJECTORY =================
        a1 = last_control_input.cpu().numpy()
        T = np.eye(4, dtype=float)
        T[:3, :3] = rpy2r(euler_init[0] + a1[0], euler_init[1], euler_init[2] + a1[1])
        T[:3, 3] = T_init[:3, 3]

        # ================= TIMING =================
        next_tick += dt
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()

        ur10_control.send_trajectory_smooth(T, dt= dt)
    
    log_file.close()

time.sleep(1)
os._exit(0)


