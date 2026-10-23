import sys
import time
import threading
import torch
from scipy.spatial.distance import cdist
import numpy as np
sys.path.append('./NatNet_SDK/samples/PythonClient')
from NatNetClient import NatNetClient 
import matplotlib.pyplot as plt
from collections import deque

class SimpleKalman6D:
    def __init__(self, dt):
        self.dt = dt

        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0

        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, i+3] = dt

        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # Ajustables
        self.Q = np.eye(6, dtype=np.float32) * 1e-1   # ruido del modelo
        self.R = np.eye(3, dtype=np.float32) * 1e-1   # ruido de medida

        self.initialized = False

    def set_dt(self, dt):
        self.dt = dt
        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, i+3] = dt

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape(3, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def step(self, z):
        if not self.initialized:
            self.x[0:3, 0] = z
            self.initialized = True

        self.predict()
        self.update(z)
        return self.x[0:3, 0].copy()


class DLOTracker_:
    def __init__(self, client_address: str, server_address: str, num_markers: int, device):

        self.client = NatNetClient()
        self.client.set_client_address(client_address)
        self.client.set_server_address(server_address)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)

        self.num_markers = 2
        self.device = device

        self.marker_positions = torch.zeros((2, 3), device=device, dtype=torch.float32)
        self.temp_cpu = torch.zeros((2, 3), dtype=torch.float32)
        self.valid = torch.zeros(2, dtype=torch.bool)

        self.client.new_frame_with_data_listener = self._on_frame

        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            ok = self.client.run("d")      # ← ya crea su propio thread
            if not ok:
                print("ERROR: Could not start NatNet client")
                raise RuntimeError("NatNet failed")

    def stop(self):
        self.running = False
        self.client.shutdown()

    def _on_frame(self, frameData):

        mocap_data = frameData["mocap_data"]
        if mocap_data is None:
            return

        labeled = mocap_data.labeled_marker_data.labeled_marker_list
        if len(labeled) == 0:
            return

        # posiciones detectadas
        detected = np.array(
            [[m.pos[0], m.pos[1], m.pos[2]] for m in labeled],
            dtype=np.float32,
        )

        # ordenar por Z
        order = np.argsort(detected[:, 2])
        detected = detected[order][:2]

        # reset
        self.valid[:] = False

        # copiar buffers
        for i in range(detected.shape[0]):
            self.temp_cpu[i] = torch.from_numpy(detected[i])
            self.valid[i] = True

        # volcar en GPU
        self.marker_positions.copy_(self.temp_cpu, non_blocking=True)

class DLOTracker_Mean:
    def __init__(self, client_address: str, server_address: str, num_markers: int, device, window_size=7):

        self.client = NatNetClient()
        self.client.set_client_address(client_address)
        self.client.set_server_address(server_address)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)

        self.num_markers = 2
        self.device = device

        self.marker_positions = torch.zeros((2, 3), device=device, dtype=torch.float32)
        self.temp_cpu = torch.zeros((2, 3), dtype=torch.float32)
        self.valid = torch.zeros(2, dtype=torch.bool)

        self.client.new_frame_with_data_listener = self._on_frame
        self.running = False
        self.last_frame = None

        # --- buffers para media ---
        self.window_size = window_size
        self.buffers = [
            [deque(maxlen=window_size) for _ in range(3)]
            for _ in range(self.num_markers)
        ]

    def start(self):
        if not self.running:
            self.running = True
            ok = self.client.run("d")
            if not ok:
                raise RuntimeError("NatNet failed")

    def stop(self):
        self.running = False
        self.client.shutdown()

    def _smooth_with_mean(self, i, pos):
        for ax in range(3):
            self.buffers[i][ax].append(pos[ax])
            pos[ax] = np.mean(self.buffers[i][ax])
        return pos

    def _on_frame(self, frameData):

        mocap_data = frameData["mocap_data"]
        if mocap_data is None:
            return
        
        # Print Timestamp
        if self.last_frame is not None and mocap_data.prefix_data.frame_number < self.last_frame:
            print("Frame number reset detected.")
           
        self.last_frame = mocap_data.prefix_data.frame_number
    
        labeled = mocap_data.labeled_marker_data.labeled_marker_list
        if len(labeled) == 0:
            return

        detected = np.array(
            [[m.pos[0], m.pos[1], m.pos[2]] for m in labeled],
            dtype=np.float32,
        )

        order = np.argsort(detected[:, 2])[::-1]  # Ordenar por Z de mayor a menor
        detected = detected[order][:2]           # Seleccionar los dos mayores
        detected = detected[np.argsort(detected[:, 2])]  # Reordenar para que el menor esté primero

        self.valid[:] = False

        for i in range(detected.shape[0]):
            smoothed = self._smooth_with_mean(i, detected[i].copy())
            self.temp_cpu[i] = torch.from_numpy(smoothed)
            self.valid[i] = True

        self.marker_positions.copy_(self.temp_cpu, non_blocking=True)



# class DLOTracker_Mean:
#     def __init__(
#         self,
#         client_address: str,
#         server_address: str,
#         device,
#         num_markers: int = 2,
#         window_size=7,
#         max_dist=0.05,   # metros
#     ):

#         # ---- NatNet ----
#         self.client = NatNetClient()
#         self.client.set_client_address(client_address)
#         self.client.set_server_address(server_address)
#         self.client.set_use_multicast(False)
#         self.client.set_print_level(0)

#         # ---- Config ----
#         self.num_markers = 2
#         self.device = device
#         self.window_size = window_size
#         self.max_dist = max_dist

#         # ---- Marker state ----
#         self.marker_positions = torch.zeros((2, 3), device=device)
#         self.temp_cpu = torch.zeros((2, 3))
#         self.valid = torch.zeros(2, dtype=torch.bool)

#         # ---- Previous state ----
#         self.prev_positions = np.zeros((2, 3), dtype=np.float32)
#         self.initialized = False

#         # ---- Moving average buffers ----
#         self.buffers = [
#             [deque(maxlen=window_size) for _ in range(3)]
#             for _ in range(2)
#         ]

#         # ---- Runtime ----
#         self.client.new_frame_with_data_listener = self._on_frame
#         self.running = False

#     def start(self):
#         if not self.running:
#             self.running = True
#             ok = self.client.run("d")
#             if not ok:
#                 raise RuntimeError("NatNet failed")

#     def stop(self):
#         self.running = False
#         self.client.shutdown()

#     # -----------------------------
#     # Smoothing
#     # -----------------------------
#     def _smooth(self, i, pos):
#         for ax in range(3):
#             self.buffers[i][ax].append(pos[ax])
#             pos[ax] = np.mean(self.buffers[i][ax])
#         return pos

#     # -----------------------------
#     # Main callback
#     # -----------------------------
#     def _on_frame(self, frameData):

#         mocap_data = frameData["mocap_data"]
#         if mocap_data is None:
#             return

#         labeled = mocap_data.labeled_marker_data.labeled_marker_list
#         if len(labeled) == 0:
#             self.valid[:] = False
#             return

#         detected = np.array(
#             [[m.pos[0], m.pos[1], m.pos[2]] for m in labeled],
#             dtype=np.float32,
#         )

#         # -----------------------------
#         # Initialization (first frame)
#         # -----------------------------
#         if not self.initialized:
#             if detected.shape[0] < 2:
#                 return
            
#             order = np.argsort(detected[:, 2])[::-1]
#             detected = detected[order][:2]         
#             detected = detected[np.argsort(detected[:, 2])]  

#             for i in range(2):
#                 pos = detected[i].copy()
#                 pos = self._smooth(i, pos)
#                 self.temp_cpu[i] = torch.from_numpy(pos)
#                 self.prev_positions[i] = pos
#                 self.valid[i] = True

#             self.marker_positions.copy_(self.temp_cpu, non_blocking=True)
#             self.initialized = True
#             return

#         # -----------------------------
#         # Tracking (track → detection)
#         # -----------------------------
#         self.valid[:] = False
#         used = np.zeros(len(detected), dtype=bool)

#         for i in range(2):
#             dists = np.linalg.norm(detected - self.prev_positions[i], axis=1)
#             j = np.argmin(dists)

#             if dists[j] < self.max_dist and not used[j]:
#                 pos = detected[j].copy()
#                 pos = self._smooth(i, pos)

#                 self.temp_cpu[i] = torch.from_numpy(pos)
#                 self.prev_positions[i] = pos
#                 self.valid[i] = True
#                 used[j] = True

#         self.marker_positions.copy_(self.temp_cpu, non_blocking=True)

        
class DLOTracker_Kalman:
    def __init__(self, client_address: str, server_address: str, num_markers: int, device):

        self.client = NatNetClient()
        self.client.set_client_address(client_address)
        self.client.set_server_address(server_address)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)

        self.num_markers = 2
        self.device = device

        self.marker_positions = torch.zeros((2, 3), device=device, dtype=torch.float32)
        self.temp_cpu = torch.zeros((2, 3), dtype=torch.float32)
        self.valid = torch.zeros(2, dtype=torch.bool)

        self.client.new_frame_with_data_listener = self._on_frame
        self.running = False

        # ---- Kalman por marcador ----
        self.filters = [SimpleKalman6D(1/180.0) for _ in range(self.num_markers)]
        self.last_time = None

    def start(self):
        if not self.running:
            self.running = True
            ok = self.client.run("d")
            if not ok:
                raise RuntimeError("NatNet failed")

    def stop(self):
        self.running = False
        self.client.shutdown()

    def _on_frame(self, frameData):

        # -------- medir dt local --------
        now = time.time()

        if self.last_time is None:
            dt = 1 / 180.0
        else:
            dt = now - self.last_time

        self.last_time = now

        # limitar dt (evitar explosiones si se bloquea)
        dt = np.clip(dt, 1e-4, 0.015)
        print("dt:", dt)

        # actualizar dt en los filtros
        for f in self.filters:
            f.set_dt(dt)

        # -------- datos mocap --------
        mocap_data = frameData["mocap_data"]
        if mocap_data is None:
            return

        labeled = mocap_data.labeled_marker_data.labeled_marker_list
        if len(labeled) == 0:
            return

        detected = np.array(
            [[m.pos[0], m.pos[1], m.pos[2]] for m in labeled],
            dtype=np.float32,
        )

        # ordenar por Z (tu mismo criterio)
        order = np.argsort(detected[:, 2])
        detected = detected[order][:2]

        self.valid[:] = False

        # ---- aplicar Kalman ----
        for i in range(detected.shape[0]):
            filtered = self.filters[i].step(detected[i])
            self.temp_cpu[i] = torch.from_numpy(filtered.astype(np.float32))
            self.valid[i] = True

        # ---- volcar a GPU ----
        self.marker_positions.copy_(self.temp_cpu, non_blocking=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = DLOTracker_Mean(client_address='169.254.126.181', server_address='169.254.126.180', num_markers=1, device=device)
    tracker.start()
    error_log = []
    
    # try:
    #     while True:

    #         if not tracker.initialized:
    #             time.sleep(0.01)
    #             continue

    #         # Esperamos a que el callback actualice datos
    #         time.sleep(0.01)

    #         # ---- calculamos error por frame ----
    #         frame_errors = []

    #         for i in range(tracker.num_markers):

    #             # estimación del Kalman
    #             est = tracker.kalman_state[i, :3]

    #             # salida publicada (lo que usas realmente)
    #             meas = tracker.marker_positions[i].cpu().numpy()

    #             e = np.linalg.norm(meas - est)

    #             frame_errors.append(e)

    #         if frame_errors:
    #             error_log.append(float(np.mean(frame_errors)))

    # except KeyboardInterrupt:
    #     print("\nParando…")
    #     tracker.stop()

    #     # ---- mostrar gráfica ----
    #     if len(error_log) == 0:
    #         print("No hay datos para graficar.")
    #         return

    #     plt.figure()
    #     plt.plot(error_log)
    #     plt.title("Error aproximado del filtro Kalman")
    #     plt.xlabel("Frame")
    #     plt.ylabel("Error (m)")
    #     plt.grid(True)
    #     plt.show()
    
    # time.sleep(1)
    # while True:
    #     print(tracker.marker_positions)
    #     # if (dloTracker.marker_positions[0, 2] > dloTracker.marker_positions[2, 2]):
    #     #     exit()
    #     time.sleep(0.05)

    marker_pos = []
    try:
        while True:

            if not tracker.running:
                time.sleep(0.01)
                continue

            # Esperamos a que el callback actualice datos
            time.sleep(0.015)

            marker_pos.append(tracker.marker_positions[0].cpu().numpy().copy())

    except KeyboardInterrupt:
        print("\nParando…")
        tracker.stop()

        marker_pos = np.array(marker_pos)
        plt.figure()
        plt.plot(marker_pos[:, 0], marker_pos[:, 1], marker='o')
        plt.title("Trayectoria del marcador 0")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        #Velocity dt = 0.05s
        plt.figure()
        velocities = np.diff(marker_pos, axis=0) / 0.015
        plt.plot(velocities)
        plt.title("Velocidades del marcador 0")
        plt.xlabel("Frame")
        plt.ylabel("Velocidad (m/s)")
        plt.grid(True)
        plt.show()

        # plot marker positions in each axis
        plt.figure()
        plt.plot(marker_pos[:, 0], label='X')
        plt.plot(marker_pos[:, 1], label='Y')
        plt.plot(marker_pos[:, 2], label='Z')
        plt.title("Posiciones del marcador en cada eje")
        plt.xlabel("Frame")
        plt.ylabel("Posición (m)")
        plt.legend()
        plt.grid(True)
        plt.show()

        for item in marker_pos:
            print(item) 

if __name__ == "__main__":
    main()