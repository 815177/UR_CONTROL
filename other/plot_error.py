import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import *


#------------------------------ REVISAR -------------------------#

def process_file(FICHERO):
    df = pd.read_csv(FICHERO)
    timestamps = df["t"].values
    pos_b = df[["p_b_x", "p_b_y", "p_b_z"]].values
    euler_b = df[["phi_b_x", "phi_b_y", "phi_b_z"]].values
    pos_b_vel = df[["dp_b_x", "dp_b_y", "dp_b_z"]].values
    euler_b_vel = df[["dphi_b_x", "dphi_b_y", "dphi_b_z"]].values
    pos_e = df[["p_e_x", "p_e_y", "p_e_z"]].values
    pos_e_vel = df[["dp_e_x", "dp_e_y", "dp_e_z"]].values

    return timestamps, euler_b, pos_e

def calcular_errores(t, ang1, ang2, pos1, pos2):
    # --- Errores instantáneos ---
    error_ang = ang1 - ang2
    error_pos = pos1 - pos2

    # --- Errores absolutos y normados ---
    error_ang_norm = np.linalg.norm(error_ang, axis=1)
    error_pos_norm = np.linalg.norm(error_pos, axis=1)

    # --- Errores medios absolutos (MAE) ---
    MAE_ang = np.mean(np.abs(error_ang), axis=0)
    MAE_pos = np.mean(np.abs(error_pos), axis=0)

    # --- RMSE ---
    RMSE_ang = np.sqrt(np.mean(error_ang**2, axis=0))
    RMSE_pos = np.sqrt(np.mean(error_pos**2, axis=0))

    # --- Errores totales (norma) ---
    MAE_ang_total = np.mean(error_ang_norm)
    MAE_pos_total = np.mean(error_pos_norm)
    RMSE_ang_total = np.sqrt(np.mean(error_ang_norm**2))
    RMSE_pos_total = np.sqrt(np.mean(error_pos_norm**2))

    # --- Error acumulado ---
    dt = np.mean(np.diff(t))
    error_ang_acum = np.cumsum(error_ang_norm) * dt
    error_pos_acum = np.cumsum(error_pos_norm) * dt

    resultados = {
        "error_ang": error_ang,
        "error_pos": error_pos,
        "error_ang_norm": error_ang_norm,
        "error_pos_norm": error_pos_norm,
        "error_ang_acum": error_ang_acum,
        "error_pos_acum": error_pos_acum,
        "MAE_ang": MAE_ang,
        "MAE_pos": MAE_pos,
        "RMSE_ang": RMSE_ang,
        "RMSE_pos": RMSE_pos,
        "MAE_ang_total": MAE_ang_total,
        "MAE_pos_total": MAE_pos_total,
        "RMSE_ang_total": RMSE_ang_total,
        "RMSE_pos_total": RMSE_pos_total
    }

    return resultados

def calcular_errores_pos(t, pos1, pos2):
    # --- Errores instantáneos ---
    error_pos = pos1[:, :2] - pos2[:, :2]

    # --- Errores absolutos y normados ---
    error_pos_norm = np.linalg.norm(error_pos, axis=1)

    # --- Errores medios absolutos (MAE) ---
    MAE_pos = np.mean(np.abs(error_pos), axis=0)

    # --- RMSE ---
    RMSE_pos = np.sqrt(np.mean(error_pos**2, axis=0))

    # --- Errores totales (norma) ---
    MAE_pos_total = np.mean(error_pos_norm)
    RMSE_pos_total = np.sqrt(np.mean(error_pos_norm**2))

    # --- Error acumulado ---
    dt = np.mean(np.diff(t))
    error_pos_acum = np.cumsum(error_pos_norm) * dt

    resultados = {
        "error_pos": error_pos,
        "error_pos_norm": error_pos_norm,
        "error_pos_acum": error_pos_acum,
        "MAE_pos": MAE_pos,
        "RMSE_pos": RMSE_pos,
        "MAE_pos_total": MAE_pos_total,
        "RMSE_pos_total": RMSE_pos_total
    }

    return resultados

def graficar_errores(t, resultados):
    # --- Gráfica de errores instantáneos ---
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, resultados["error_pos_norm"], label="Error posición (norma)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error posición [m]")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, resultados["error_ang_norm"], label="Error ángulo (norma)", color='orange')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error ángulo [rad]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Gráfica de error acumulado ---
    plt.figure(figsize=(12, 6))
    plt.plot(t, resultados["error_pos_acum"], label="Error acumulado posición")
    plt.plot(t, resultados["error_ang_acum"], label="Error acumulado ángulo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error acumulado")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_errores_pos(t, resultados):
    # --- Gráfica de errores instantáneos ---
    plt.plot(t, resultados["error_pos_norm"], label="Error posición (norma)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error posición [m]")
    plt.grid()
    plt.legend()
    plt.show()

    # --- Gráfica de error acumulado ---
    plt.figure(figsize=(12, 6))
    plt.plot(t, resultados["error_pos_acum"], label="Error acumulado posición")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error acumulado")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def generar_trayectoria(form: str, duracion=20, dt=0.005, velocidad=1, 
                        centro=(0,0), radio=1, escala=1):

    t = np.arange(0, duracion, dt)
    w = 2 * np.pi * velocidad  
    y0, z0 = centro

    if form.lower() == 'circulo':
        y = y0 + radio * np.cos(w * t)
        z = z0 + radio * np.sin(w * t)

    elif form.lower() == 'corazon':
        theta = w * t
        y = y0 + escala * 16 * (np.sin(theta))**3 / 17
        z = z0 + escala * (13*np.cos(theta) - 5*np.cos(2*theta) -
                           2*np.cos(3*theta) - np.cos(4*theta)) / 17

    elif form.lower() == 'infinito':
        theta = w * t
        a = escala
        y = y0 + a * np.sin(theta)
        z = z0 + a * np.sin(theta) * np.cos(theta)
    
    elif form.lower() == 'cuadrado':
        ciclos = int(duracion * velocidad)
        puntos_por_ciclo = int(len(t) / max(ciclos, 1))

        y = np.full(len(t), y0)
        z = np.full(len(t), z0)

        for i in range(ciclos):
            ini = i * puntos_por_ciclo
            fin = (i + 1) * puntos_por_ciclo if (i + 1) * puntos_por_ciclo <= len(t) else len(t)
            n = fin - ini
            seg = n // 4

            y_temp = np.zeros(n)
            z_temp = np.zeros(n)

            # Escala aplicada al tamaño
            lado = radio * escala

            # Lado 1: de izquierda a derecha (parte inferior)
            y_temp[:seg] = np.linspace(y0 - lado, y0 + lado, seg)
            z_temp[:seg] = z0 - lado

            # Lado 2: de abajo hacia arriba (lado derecho)
            y_temp[seg:2*seg] = y0 + lado
            z_temp[seg:2*seg] = np.linspace(z0 - lado, z0 + lado, seg)

            # Lado 3: de derecha a izquierda (parte superior)
            y_temp[2*seg:3*seg] = np.linspace(y0 + lado, y0 - lado, seg)
            z_temp[2*seg:3*seg] = z0 + lado

            # Lado 4: de arriba hacia abajo (lado izquierdo)
            y_temp[3*seg:] = y0 - lado
            z_temp[3*seg:] = np.linspace(z0 + lado, z0 - lado, n - 3*seg)

            # Asignar al tramo correspondiente
            y[ini:fin] = y_temp
            z[ini:fin] = z_temp

    elif form.lower() in ['linea_y', 'linea_z', 'diagonal']:
        ciclos = int(duracion * velocidad)
        t_ciclo = np.linspace(0, 1, int(len(t) / max(ciclos, 1)))
        saw = np.abs(2 * (t_ciclo % 1) - 1)
        saw_full = np.tile(saw, ciclos + 1)[:len(t)]
        amp = radio * escala

        if form.lower() == 'linea_y':
            y = y0 - amp + 2 * amp * saw_full
            z = np.full_like(y, z0)

        elif form.lower() == 'linea_z':
            y = np.full_like(t, y0)
            z = z0 - amp + 2 * amp * saw_full

        elif form.lower() == 'diagonal':
            y = y0 - amp + 2 * amp * saw_full
            z = z0 - amp + 2 * amp * saw_full

    else:
        raise ValueError("Forma no reconocida. Usa: 'circulo', 'corazon', 'infinito', 'cuadrado', 'linea_y', 'linea_z' o 'diagonal'.")

    return t, y, z


mode = argv[1]

if (mode == '0'):

    FICHERO1 = argv[2]
    FICHERO2 = argv[3]

    t1, ang1, pos1 = process_file(FICHERO1)
    t2, ang2, pos2 = process_file(FICHERO2)

    n = min(len(t1), len(t2))
    t = t1[:n]
    ang1, ang2 = ang1[:n], ang2[:n]
    pos1, pos2 = pos1[:n], pos2[:n]

    resultados = calcular_errores(t, ang1, ang2, pos1, pos2)

    print("\n==== RESULTADOS DE ERROR ====")
    print(f"MAE posición [x,y,z]: {resultados['MAE_pos']}")
    print(f"RMSE posición [x,y,z]: {resultados['RMSE_pos']}")
    print(f"MAE total posición: {resultados['MAE_pos_total']:.6f}")
    print(f"RMSE total posición: {resultados['RMSE_pos_total']:.6f}\n")

    print(f"MAE ángulo [x,y,z]: {resultados['MAE_ang']}")
    print(f"RMSE ángulo [x,y,z]: {resultados['RMSE_ang']}")
    print(f"MAE total ángulo: {resultados['MAE_ang_total']:.6f}")
    print(f"RMSE total ángulo: {resultados['RMSE_ang_total']:.6f}")

    graficar_errores(t, resultados)

else:

    FICHERO2 = argv[2]
    t2, ang2, pos2 = process_file(FICHERO2)

    t1, x, y = generar_trayectoria('circulo', velocidad=1.25, radio=0.25, centro=(1.0, 0.9), dt=0.05) 
    # t1, x, y = generar_trayectoria('corazon', velocidad=0.8, escala=0.2, centro=(1.0,0.9), dt=0.05) 
    # t1, x, y = generar_trayectoria('cuadrado', velocidad=0.8, escala=0.2, centro=(1.0,0.9), dt=0.05) 
    # t1, x, y = generar_trayectoria('infinito', velocidad=0.8, escala=0.2, centro=(1.0,0.9), dt=0.05)
    # t1, x, y = generar_trayectoria('linea_y', velocidad=1, escala=0.4, centro=(1, 0.9), dt=0.05)
    # t1, x, y = generar_trayectoria('linea_z', velocidad=1, escala=0.3, centro=(1, 0.9), dt=0.05)
    # t1, x, y = generar_trayectoria('diagonal', velocidad=1, escala=0.35, centro=(1, 0.9), dt=0.05)
    zeros = np.zeros(len(t1))
    pos1 = np.column_stack((x, y, zeros))
   
    n = min(len(t1), len(t2))
    t = t1[:n]
    pos1, pos2 = pos1[:n], pos2[:n]

    # --- Gráfica de error acumulado ---
    plt.figure(figsize=(12, 6))
    plt.plot(pos1[:, 0], pos1[:, 1], label="Goal")
    plt.plot(pos2[:, 0], pos2[:, 1], label="MPC")
    plt.xlabel("Posición X [m]")
    plt.ylabel("Posición Y [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t, pos1[:, 0], label="Goal X")
    plt.plot(t, pos1[:, 1], label="Goal Y")
    plt.plot(t, pos2[:, 0], label="MPC X")
    plt.plot(t, pos2[:, 1], label="MPC Y")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Posición [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    resultados = calcular_errores_pos(t, pos1, pos2)

    print("\n==== RESULTADOS DE ERROR ====")
    print(f"MAE posición [y,z]: {resultados['MAE_pos']}")
    print(f"RMSE posición [y,z]: {resultados['RMSE_pos']}")
    print(f"MAE total posición: {resultados['MAE_pos_total']:.6f}")
    print(f"RMSE total posición: {resultados['RMSE_pos_total']:.6f}\n")

    graficar_errores_pos(t, resultados)