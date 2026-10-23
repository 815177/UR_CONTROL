
# ---------- CONFIG ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import *

FICHERO = argv[1]

def main():
    # Leer el CSV
    df = pd.read_csv(FICHERO)

    # Extraer columnas
    timestamps = df["t"].values

    pos_b = df[["p_b_x", "p_b_y", "p_b_z"]].values
    euler_b = df[["phi_b_x", "phi_b_y", "phi_b_z"]].values
    
    pos_b_vel = df[["dp_b_x", "dp_b_y", "dp_b_z"]].values
    euler_b_vel = df[["dphi_b_x", "dphi_b_y", "dphi_b_z"]].values
    
    pos_e = df[["p_e_x", "p_e_y", "p_e_z"]].values
    pos_e_vel = df[["dp_e_x", "dp_e_y", "dp_e_z"]].values

    pitch_yaw = df[["u_pitch", "u_yaw"]].values

    # ---------- PLOTS ----------

    # Posición base (x vs y)
    plt.figure(figsize=(6,6))
    plt.plot(pos_b[:,0], pos_b[:,1], label="Base")
    plt.plot(pos_e[:,0], pos_e[:,1], label="Extremo")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.title("Trayectoria 2D")
    plt.legend()
    plt.grid(True)

    # Posiciones vs tiempo
    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["x","y","z"]):
        plt.plot(timestamps, pos_b[:,i], label=f"Base {lbl}")
        plt.plot(timestamps, pos_e[:,i], "--", label=f"Extremo {lbl}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Posición [m]")
    plt.title("Posiciones")
    plt.legend()
    plt.grid(True)

    # Euler vs tiempo
    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["roll","pitch","yaw"]):
        plt.plot(timestamps, euler_b[:,i], label=lbl)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Ángulo [rad]")
    plt.title("Orientación (Euler)")
    plt.legend()
    plt.grid(True)

    # Velocidades lineales (base)
    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["x","y","z"]):
        plt.plot(timestamps, pos_b_vel[:,i], label=f"v_{lbl}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [m/s]")
    plt.title("Velocidades lineales (base)")
    plt.legend()
    plt.grid(True)


    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["x","y","z"]):
        plt.plot(timestamps, pos_e_vel[:,i], label=f"v_{lbl}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [m/s]")
    plt.title("Velocidades lineales (extremo)")
    plt.legend()
    plt.grid(True)

    # Velocidades angulares (Euler)
    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["roll","pitch","yaw"]):
        plt.plot(timestamps, euler_b_vel[:,i], label=f"d{lbl}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [rad/s]")
    plt.title("Velocidades angulares")
    plt.legend()
    plt.grid(True)

    # Acciones (pitch / yaw)
    plt.figure(figsize=(10,6))
    for i, lbl in enumerate(["pitch","yaw"]):
         plt.plot(timestamps, pitch_yaw[:,i], label=f"u_{lbl}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Input [º]")
    plt.title("Actions")
    plt.legend()
    plt.grid(True)

    plt.show()
    

if __name__ == "__main__":
    main()
