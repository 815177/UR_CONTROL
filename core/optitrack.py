import sys
import time
import socket
import threading
import torch
from scipy.spatial.distance import cdist
import numpy as np
sys.path.append('./NatNet_SDK/samples/PythonClient')
from NatNetClient import NatNetClient, get_message_id
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

# ---------------------------------------------------------------------------
# Cliente NatNet que siempre procesa el frame más reciente
# ---------------------------------------------------------------------------
class LatestFrameNatNetClient(NatNetClient):
    """
    El hilo de datos del SDK hace un recvfrom bloqueante y desempaqueta TODOS los paquetes en orden
    (en Python puro). Si el desempaquetado va más lento que el OptiTrack (~180 Hz), p.ej. porque el
    GIL está ocupado con torch/rosbridge, los paquetes se acumulan en el buffer del socket y los datos
    llegan con un retraso que se mantiene (en el dataset: saltos entre ~50 ms y ~225 ms).
    Aquí, tras cada recvfrom, se vacía el socket y solo se desempaqueta el último frame de datos.
    """
    def __init__(self):
        super().__init__()
        self.frames_skipped = 0              # frames descartados por llegar con retraso

    # Sustituye a NatNetClient.__data_thread_function (name mangling)
    def _NatNetClient__data_thread_function(self, in_socket, stop, gprint_level):
        recv_buffer_size = 128 * 1024
        timeout = in_socket.gettimeout()
        while not stop():
            try:
                data, _ = in_socket.recvfrom(recv_buffer_size)
            except socket.timeout:
                continue
            except OSError as msg:
                if not stop():
                    print("ERROR: data socket access error occurred:\n  %s" % msg)
                    return 1
                continue

            latest = None
            in_socket.setblocking(False)
            try:
                while True:
                    if len(data) > 0:
                        if get_message_id(data) == self.NAT_FRAMEOFDATA:
                            if latest is not None:
                                self.frames_skipped += 1
                            latest = data
                        else:
                            self._NatNetClient__process_message(data, 0)
                    data, _ = in_socket.recvfrom(recv_buffer_size)
            except (BlockingIOError, InterruptedError):
                pass
            except OSError as msg:
                if not stop():
                    print("ERROR: data socket access error occurred:\n  %s" % msg)
            finally:
                in_socket.settimeout(timeout)

            if latest is not None:
                self._NatNetClient__process_message(latest, gprint_level())
        return 0


# ---------------------------------------------------------------------------
# Asignación robusta de marcadores (base / punta)
# ---------------------------------------------------------------------------
class TwoMarkerAssigner:
    """
    Mantiene la identidad de los dos marcadores entre frames: índice 0 = punta (p_e), 1 = base (p_b).

    - Inicialización: la base es el marcador más alto en Z o el más cercano a `init_base_position` si se
      da. La regla de Z solo es válida con el cable colgando en reposo, así que se exigen `init_frames`
      frames con los marcadores quietos (< `init_still` m entre frames). Si no se da `cable_length`, solo
      cuentan los frames con exactamente 2 marcadores (los reflejos intermitentes no rompen la inicialización).
    - Seguimiento: entre todas las hipótesis (par punta/base, o solo uno visible) se escoge la de menor
      distancia a la posición predicha (velocidad constante), con puertas de distancia y la restricción
      de longitud del cable |punta - base| ~ L. La ordenación por un eje no interviene, así que no se
      intercambian aunque cambie su orden en Z.
    - Un marcador que no se encuentra se marca como no válido y conserva su última posición; nunca se
      rellena con el otro marcador.
    """
    END, BASE = 0, 1

    def __init__(self, cable_length=None, length_tol=0.05, max_speed=(8.0, 1.0), gate_margin=0.02,
                 max_missing_s=0.25, init_base_position=None, init_frames=10, init_still=0.005, base_radius=None):
        self.L = cable_length                 # None -> se mide en la inicialización
        self.length_tol = length_tol          # m
        self.max_speed = np.asarray(max_speed, dtype=float)   # m/s (punta, base)
        self.gate_margin = gate_margin        # m
        self.max_missing_s = max_missing_s    # s sin ver un marcador -> se reinicia su puerta
        self.init_base_position = None if init_base_position is None else np.asarray(init_base_position, float)
        self.init_frames = init_frames
        self.init_still = init_still
        self.base_radius = base_radius        # opcional: la base no se aleja más de esto de su posición inicial
        self.reset()

    def reset(self):
        self.pos = np.zeros((2, 3))
        self.vel = np.zeros((2, 3))
        self.t_seen = np.full(2, -np.inf)
        self.valid = np.zeros(2, dtype=bool)
        self.initialized = False
        self._init_count = 0
        self._init_prev = None
        self.base_anchor = None

    # --- inicialización ---
    def _init_pair(self, P):
        if len(P) < 2:
            return None
        if self.init_base_position is not None:
            b = int(np.argmin(np.linalg.norm(P - self.init_base_position, axis=1)))
            d = np.linalg.norm(P - P[b], axis=1); d[b] = -1
            if self.L is not None:
                d = np.where(d < 0, np.inf, np.abs(d - self.L))
                e = int(np.argmin(d))
            else:
                e = int(np.argmax(d))
        else:
            top2 = np.argsort(P[:, 2])[::-1][:2]
            b, e = int(top2[0]), int(top2[1])
        if self.L is not None and abs(np.linalg.norm(P[e] - P[b]) - self.L) > self.length_tol:
            return None
        return P[e], P[b]

    def _try_initialize(self, P, t):
        if self.L is None and len(P) != 2:
            return                            # sin longitud conocida: solo frames con exactamente 2 marcadores (sin reflejos)
        pair = self._init_pair(P)
        if pair is None:
            self._init_count, self._init_prev = 0, None
            return
        pair = np.stack(pair)
        if self._init_prev is not None and np.linalg.norm(pair - self._init_prev, axis=1).max() > self.init_still:
            self._init_count = 0
        self._init_prev = pair
        self._init_count += 1
        if self._init_count >= self.init_frames:
            self.pos[:] = pair
            self.vel[:] = 0
            self.t_seen[:] = t
            self.valid[:] = True
            if self.L is None:
                self.L = float(np.linalg.norm(pair[0] - pair[1]))
            self.base_anchor = pair[1].copy()
            self.initialized = True

    # --- seguimiento ---
    def update(self, P, t):
        """P: (n, 3) posiciones de marcadores no ocluidos del frame; t: tiempo del frame (s)."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        if not self.initialized:
            self._try_initialize(P, t)
            return self.pos.copy(), self.valid.copy()

        dt = np.clip(t - self.t_seen, 1e-3, None)
        lost = dt > self.max_missing_s
        pred = self.pos + self.vel * np.minimum(dt, self.max_missing_s)[:, None]
        gate = self.max_speed * np.minimum(dt, self.max_missing_s) + self.gate_margin
        gate[lost] = np.inf                   # perdido: cualquier candidato que cumpla la longitud

        d = np.linalg.norm(P[:, None, :] - pred[None, :, :], axis=2)   # (n, 2)
        ok = d <= gate[None, :]
        if self.base_radius is not None and self.base_anchor is not None:
            ok[:, self.BASE] &= np.linalg.norm(P - self.base_anchor, axis=1) <= self.base_radius

        best, best_cost = (None, None), np.inf
        n = len(P)
        for e in range(n):                    # hipótesis con los dos marcadores
            for b in range(n):
                if e == b or not (ok[e, self.END] and ok[b, self.BASE]):
                    continue
                len_err = abs(np.linalg.norm(P[e] - P[b]) - self.L)
                if len_err > self.length_tol:
                    continue
                cost = d[e, self.END] + d[b, self.BASE] + len_err
                if cost < best_cost:
                    best, best_cost = (e, b), cost
        if best == (None, None):              # solo uno visible (sin perder la identidad del otro)
            for i, other in ((self.END, self.BASE), (self.BASE, self.END)):
                for j in range(n):
                    if not ok[j, i] or lost[i]:
                        continue
                    # si el otro marcador está bien localizado, exigir la longitud del cable
                    if not lost[other] and abs(np.linalg.norm(P[j] - pred[other]) - self.L) > self.length_tol + gate[other]:
                        continue
                    cost = d[j, i] + 1.0      # penaliza frente a las hipótesis de par
                    if cost < best_cost:
                        best = (j, None) if i == self.END else (None, j)
                        best_cost = cost

        self.valid[:] = False
        for i, j in ((self.END, best[0]), (self.BASE, best[1])):
            if j is None:
                continue
            new = P[j]
            if np.isfinite(self.t_seen[i]) and not lost[i]:
                v = (new - self.pos[i]) / dt[i]
                self.vel[i] = 0.5 * self.vel[i] + 0.5 * v
            else:
                self.vel[i] = 0
            self.pos[i] = new
            self.t_seen[i] = t
            self.valid[i] = True
        return self.pos.copy(), self.valid.copy()


class DLOTracker_Mean:
    """
    Tracker de los dos marcadores del DLO. marker_positions[0] = punta (p_e), marker_positions[1] = base (p_b).

    Cambios respecto a la versión anterior:
      - Siempre usa el frame más reciente del OptiTrack (LatestFrameNatNetClient) -> sin retraso acumulado.
      - Identidad de marcadores por seguimiento + longitud del cable (TwoMarkerAssigner), no por orden en Z.
      - Descarta marcadores ocluidos y nunca mezcla los buffers de la media entre marcadores.
      - Guarda número de frame, timestamp de Motive y hora de recepción (get_state) para poder medir y
        registrar la latencia.
    """
    def __init__(self, client_address: str, server_address: str, num_markers: int, device, window_size=7,
                 **assigner_kwargs):

        self.client = LatestFrameNatNetClient()
        self.client.set_client_address(client_address)
        self.client.set_server_address(server_address)
        self.client.set_use_multicast(False)
        self.client.set_print_level(0)

        self.num_markers = 2
        self.device = device

        self.marker_positions = torch.zeros((2, 3), device=device, dtype=torch.float32)
        self.temp_cpu = torch.zeros((2, 3), dtype=torch.float32)
        self.valid = torch.zeros(2, dtype=torch.bool)

        self.assigner = TwoMarkerAssigner(**assigner_kwargs)
        self.lock = threading.Lock()
        self.frame_number = -1
        self.mocap_timestamp = float('nan')   # s, reloj de Motive
        self.t_recv = float('nan')            # s, time.perf_counter() al procesar el frame

        self.client.new_frame_with_data_listener = self._on_frame
        self.running = False
        self.last_frame = None

        # --- buffers para media (uno por identidad de marcador) ---
        self.window_size = window_size
        self.buffers = [deque(maxlen=window_size) for _ in range(self.num_markers)]

    def start(self):
        if not self.running:
            self.running = True
            ok = self.client.run("d")
            if not ok:
                raise RuntimeError("NatNet failed")

    def stop(self):
        self.running = False
        self.client.shutdown()

    def get_state(self):
        """Copia coherente: posiciones (2,3) np, válidos, nº de frame, timestamp Motive, edad del dato (s)."""
        with self.lock:
            return {
                "positions": self.temp_cpu.numpy().copy(),
                "valid": self.valid.numpy().copy(),
                "frame_number": self.frame_number,
                "mocap_timestamp": self.mocap_timestamp,
                "age": time.perf_counter() - self.t_recv,
                "frames_skipped": self.client.frames_skipped,
            }

    def _on_frame(self, frameData):
        t_recv = time.perf_counter()
        mocap_data = frameData["mocap_data"]
        if mocap_data is None:
            return

        frame_number = mocap_data.prefix_data.frame_number
        if self.last_frame is not None and frame_number < self.last_frame:
            print("Frame number reset detected.")
            self.assigner.reset()
            for b in self.buffers:
                b.clear()
        self.last_frame = frame_number

        timestamp = frameData.get("timestamp", -1)
        t_frame = timestamp if timestamp is not None and timestamp >= 0 else t_recv

        labeled = mocap_data.labeled_marker_data.labeled_marker_list
        detected = np.array(
            [[m.pos[0], m.pos[1], m.pos[2]] for m in labeled if not (m.param & 0x01)],   # sin ocluidos
            dtype=np.float32,
        ).reshape(-1, 3)

        pos, valid = self.assigner.update(detected, t_frame)

        with self.lock:
            for i in range(self.num_markers):
                if valid[i]:
                    self.buffers[i].append(pos[i])
                    self.temp_cpu[i] = torch.from_numpy(np.mean(self.buffers[i], axis=0).astype(np.float32))
            self.valid[:] = torch.from_numpy(valid)
            self.frame_number = frame_number
            self.mocap_timestamp = t_frame
            self.t_recv = t_recv
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