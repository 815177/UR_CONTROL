# Informe: diferencia train/val en la red DLO con datos reales

Rama `regularizacion`. Todo lo de abajo se ha medido sobre los datos del repo (`datasets/dlo`, 50 trayectorias de 630 pasos a 20 Hz; el entrenamiento usa las 34 de `train_simple.py`).

## 1. Resumen

1. **La causa principal no es la red: es el retardo del OptiTrack.** El retardo entre el mocap y el robot no es constante. Va de ~25 a ~290 ms según la trayectoria y a veces salta en mitad de una grabación (de ~60 a ~225 ms). Lo provoca el cliente NatNet, que se queda atrás procesando paquetes. La red no puede saber qué retardo tiene cada trayectoria, así que en trayectorias que no ha visto predice desplazada en el tiempo.
2. **Con el split 80/20 temporal, al alinear el mocap la diferencia entre train y val desaparece:** 2.09 cm frente a 2.10 cm, cuando antes era de ~1.9×.
3. **El problema real es generalizar a trayectorias nuevas,** que es lo que ve el MPC y lo que tú ves en `test.py`. Ahí el error pasa de **~13.9 cm a ~3.2 cm**:
   - alinear el mocap aporta un factor 3;
   - cambiar el encoder MLP de un frame por un **encoder GRU sobre los últimos 10 frames** aporta otro 30%.
4. **El código del OptiTrack se ha reescrito.** Ahora procesa siempre el frame más reciente, así que no acumula retraso. La identidad base/punta se mantiene por seguimiento y longitud del cable, no por orden en Z. Descarta los marcadores ocluidos y registra la edad de cada dato.

## 2. Qué había en los datos

### 2.1 Comprobaciones que salen bien
- **Paso de tiempo estable:** dt = 50.0 ± 0.7 ms, sin huecos.
- **Sin NaN ni frames repetidos.**
- **Sin intercambios de marcadores:** la distancia base–punta es siempre 0.69–0.72 m.
- **Validación dentro del rango de entrenamiento:** el último 20% no se sale del rango del primer 80% (0% de muestras fuera, tanto en `p_e` como en `u`). Un modelo lineal (ridge) no muestra ninguna diferencia train/val con tu split (1.03×). Por tanto, la validación no es "más difícil".

### 2.2 Retardo comando → robot: constante (se puede aprender)
- `u` → orientación medida (`/joint_states` + FK): **4 pasos (~200 ms) en todas las trayectorias**, con una amplitud de ~0.8 de la comandada.
- Se explica por el `time_from_start = 0.1 s`, el filtro `alpha = 0.8` de `send_trajectory_smooth` y rosbridge. Como es constante, la red lo puede aprender.

### 2.3 Retardo robot → mocap: variable (no se puede aprender)
El marcador de la base va rígido al gripper, así que su posición es una función conocida de la orientación del robot: `p_b(t) = c + R(phi(t − τ)) · o`. Ajustando la geometría del marcador (`o ≈ [0.123, 0.018, −0.008]` m, común a todo el dataset) se obtiene el retardo τ con un error de ajuste de 4–14 mm:

- **Entre trayectorias:** τ va de 0.5 a 5.75 pasos (25–290 ms). Se agrupa en dos estados, ~1 paso (~50 ms) y ~4.5 pasos (~225 ms).
- **Dentro de una trayectoria:** en las trayectorias 19, 22, 26, 29, 33 y 43 el retardo salta.
  - **`traj43`:** 1.25 pasos durante los primeros ~24 s y luego 3.25 → 4.5 en los últimos ~6 s, que son justo tu 20% de validación. Por eso, al meterla, el error salta a 2.7 cm.
  - **Otros casos:** la 19 baja de 4.75 a 0.75, la 22 sube de 0.75 a 4.75 y la 33 baja de 4 a 0.75.
- **A la velocidad del extremo** (p99 ≈ 2.9 m/s), 1 paso de error de sincronización son hasta ~15 cm de error de posición.

**Por qué ocurre.** El hilo de datos del `NatNetClient` del SDK hace `recvfrom` bloqueante y desempaqueta **todos** los paquetes en orden, en Python puro (el OptiTrack va a ~180 Hz). Cuando el GIL está ocupado (torch, rosbridge/twisted), se acumulan paquetes en el buffer del socket. A partir de ahí el tracker entrega datos *viejos* con un retraso que se mantiene. Además, `DLOTracker_Mean` hacía una media de 7 frames (retraso fijo de ~17 ms) y el logger un EMA α=0.7 (~0.4 pasos). Estos dos son fijos y se pueden aprender; el de la cola no.

### 2.4 Lo que ves en `test.py` ("bien pero desplazada")
`test.py` evalúa en las trayectorias 27–34, que no están en el entrenamiento y tienen un retardo bajo (~1 paso). Con el modelo original, a 8 s de horizonte:

- error medio **13.7 cm**, frente a 1.7 cm en las trayectorias de entrenamiento;
- el **desfase temporal** de la predicción en cada trayectoria correlaciona **0.80** con el retardo del mocap de esa trayectoria;
- corrigiendo solo ese desfase, el error baja de 13.7 a 7.9 cm;
- en las trayectorias de entrenamiento el desfase es 0: la red ha memorizado el retardo de cada una.

Así que no era solo falta de trayectorias: hay un desfase temporal real en los datos.

### 2.5 Otros problemas encontrados en el repo
- **49 de 50 CSV**, `README.txt`, `DataDescriptions.py`, `NatNetCAPI.h` y `makefile` empiezan con 70 KB–1 MB de bytes nulos (`\0`). Con eso, `pd.read_csv` no encuentra la columna `t` y `import DataDescriptions` falla. Para los análisis usé copias limpias, sin tocar el repo. Revisa tu copia local y vuelve a subir los ficheros buenos.
- **Checkpoint mal descrito:** el `train_simple.py` original guardaba `hidden_dims: [64, 64]` en el checkpoint pero entrenaba con `[32, 32]`, así que el checkpoint no se podía cargar tal cual. Corregido.
- **`r2_score(pred, true)` tenía los argumentos al revés.** Corregido.
- **El `dropout=0.2` de `nn.GRU` con 1 capa no hacía nada.** Ahora solo se aplica con más de 1 capa.
- **El MPC ejecutaba con `model.train()`, así que el dropout estaba activo al optimizar.** Ahora pone en `eval` el encoder y el decoder, y deja solo el GRU en `train`, que cuDNN necesita para el backward.
- **`test.py` no llamaba a `model.eval()`.** Corregido.

## 3. Experimentos con la red

**Protocolo:**
- **Validación:** trayectorias enteras no vistas (1/5 de las 34) en **3 particiones distintas** (semillas 123, 7 y 99).
- **Métrica:** RMSE 3D en cm sobre el horizonte de 41 pasos (2 s), en modo eval.
- **Ojo al comparar:** el "RMSE" de `evaluate_and_plot` es por coordenada, es decir, este valor / √3.
- **Entrenamiento:** CPU, early stopping con la val loss de horizonte completo.

### 3.1 Resultados principales (media de 3 particiones)

| Configuración | val (cm) | val a 2 s (cm) | train (cm) |
|---|---|---|---|
| Original: MLP de 1 frame, latente 160, sin alinear | 13.99* | 19.31* | 1.86 |
| MLP regularizado (latente 64, dropout 0.15, ruido 0.05, wd 1e-4), sin alinear | 13.86 | 17.40 | 2.95 |
| + alineado v1 (un retardo constante por trayectoria) | 9.61 | 10.74 | 2.88 |
| + **alineado v2** (retardo variable en el tiempo, geometría fija) | 4.57 | 5.23 | 2.26 |
| Original (latente 160, poca reg.) + alineado v2 | 4.22 | 4.95 | 1.56 |
| **Encoder GRU (10 frames)** + alineado v2 | 3.53 | 4.06 | 1.79 |
| **Encoder GRU (10 frames) + poca reg.** (dropout 0.05, ruido 0.02, wd 1e-5) + alineado v2 | **3.15** | **3.72** | 1.40 |
| Encoder GRU 10 + comandos en el encoder + alineado v2 | 3.56 | 4.12 | 1.76 |
| Encoder GRU 10 + comandos + poca reg. + alineado v2 | 3.16 | 3.80 | 1.40 |

\* Una sola partición.

**Partición temporal 80/20 (la tuya):**

| Configuración | train (cm) | val (cm) | val/train |
|---|---|---|---|
| Original (latente 160, poca reg.), sin alinear | 1.79 | 2.15 | 1.20 |
| MLP regularizado, sin alinear | 2.53 | 3.03 | 1.20 |
| MLP regularizado + alineado v2 | 2.09 | 2.10 | **1.00** |
| Encoder GRU 10 + comandos + alineado v2 | 1.83 | 1.75 | 0.96 |
| Encoder GRU 10 + comandos + poca reg. + alineado v2 | 1.41 | 1.35 | 0.96 |

### 3.2 Otras variantes (partición 123, alineado v2; referencia MLP regularizado = 4.16 cm)

| Variante | val (cm) | Conclusión |
|---|---|---|
| Encoder MLP con 5 frames concatenados | 3.77 | La historia ayuda, aunque menos que con un GRU |
| Encoder GRU con 20 frames | 2.81 | Parecido a 10 frames (3.07 en esta partición); 10 basta |
| Latente 32 | 3.76 | Más pequeño no empeora |
| Latente 128 | 4.55 | Peor |
| GRU de dinámica de 2 capas | 4.64 | Peor |
| Salida residual (predecir `p_e − p_e(t0)`) | 5.28–5.53 | No ayuda (sin alinear: 11.4) |
| MLP de 6 frames + comandos | 3.64 | Pequeña mejora |
| Sin alinear + MLP de 6 frames + comandos | 11.88 | Sin alinear nada compensa el retardo |
| Las 50 trayectorias | 4.02 / 3.28 (GRU) | No comparable directamente: la validación incluye trayectorias excluidas. No mejora de forma clara |

### 3.3 Conclusiones sobre la red
1. **Lo que más importa es la calidad temporal de los datos.** Ningún cambio de arquitectura compensa un retardo de sensor que cambia entre trayectorias.
2. **Un encoder GRU sobre la historia es mejor que un MLP sobre un frame.** Las velocidades medidas son ruidosas y el estado del cable (oscilación) depende de la historia.
3. **Con datos limpios sobra regularización fuerte.** Lo que tomé al principio por sobreajuste era en gran parte el retardo. Mi primer commit (más dropout, latente 64) reducía la diferencia train/val pero **empeoraba** el error de validación (1.49 → 1.82 cm por coordenada con tu split). Con el encoder GRU, la configuración suave es la mejor.
4. **Más capas o más latente no ayudan** con 34 trayectorias.
5. **Comandos en el encoder:** con ~200 ms de retardo comando → robot, en `t0` hay ~4 comandos enviados que aún no se reflejan en el estado medido, así que tiene sentido físico darlos al encoder. En una partición bajó a 2.21–2.63 cm, pero la media de las 3 particiones no mejora: 3.56 frente a 3.53 con regularización normal y 3.16 frente a 3.15 con la suave. Queda implementado (`ENC_COMMANDS`) pero desactivado. Con datos sin retardo variable conviene volver a probarlo.

### 3.4 Resultado final (el `train_simple.py` de la rama, ejecutado completo)

Configuración: `ALIGN_MOCAP=True`, `ENC_HISTORY=10` (encoder GRU), `ENC_COMMANDS=False`, latente 64, `DROPOUT=0.05`, `NOISE_STD=0.02`, `WEIGHT_DECAY=1e-5`. Parada temprana en la época 116 (el mejor modelo es el de la 96); ~10 min en CPU. El modelo tiene 37k parámetros, frente a 92k del original.

**Con tu métrica (`evaluate_and_plot`, RMSE por coordenada) y tu split 80/20:**

| | MSE train | MSE val | val/train | RMSE val | R² val |
|---|---|---|---|---|---|
| Original | 1.19e-4 | 2.22e-4 | 1.87× | 1.49 cm | 0.989 |
| **Final** | 6.4e-5 | 5.9e-5 | **0.91×** | **0.77 cm** | 0.997 |

**Trayectorias no vistas 27–34 a 8 s** (lo que hace `test.py`; RMSE 3D):

| Trayectoria | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | media |
|---|---|---|---|---|---|---|---|---|---|
| Original | 11.48 | 5.01 | 17.32 | 3.45 | 24.96 | 5.99 | 12.69 | 28.33 | **13.65** |
| Final (mocap alineado) | 2.82 | 1.64 | 2.49 | 1.13 | 9.62 | 0.75 | 13.50 | 5.82 | **4.72** |
| Final (mocap sin alinear) | 5.54 | 4.18 | 4.15 | 3.86 | 18.60 | 9.39 | 26.61 | 12.54 | 10.61 |

- **Las trayectorias 31 y 33 son las de peor calidad del dataset.** La 33 tiene un salto de retardo dentro de la grabación y el mayor residuo del ajuste rígido (13.9 mm). La 31 tiene deriva y 220 frames con residuo superior a 2 cm.
- **Sin alinear, la predicción del modelo final sale desfasada justo el retardo del mocap** (+1 a +4 pasos). El modelo predice ahora en el tiempo del robot. En el sistema real eso exige un tracker con latencia baja y constante, que es lo que hace el tracker nuevo.

## 4. Cambios en el código (rama `regularizacion`)

| Fichero | Cambio |
|---|---|
| `core/optitrack.py` | `LatestFrameNatNetClient`: tras cada `recvfrom` vacía el socket y solo desempaqueta el último frame. `TwoMarkerAssigner`: identidad punta/base por seguimiento (predicción a velocidad constante, puertas de distancia y longitud del cable), inicialización con el cable quieto, sin mezclar buffers, descarte de ocluidos (`param & 1`), reacquisición tras pérdidas. `DLOTracker_Mean` mantiene la misma interfaz (`marker_positions[0]` = punta, `[1]` = base) y añade `get_state()` (frame, timestamp de Motive, edad del dato, frames descartados). |
| `controllers/gen_dataset_smooth.py`, `core/ur_control.py` | El CSV registra `mocap_frame`, `mocap_age`, `mocap_valid_e`, `mocap_valid_b` y `robot_age`, para medir y corregir el retardo sin estimarlo. |
| `training/align.py` | Estima τ(t) por ventanas con la geometría del marcador fija y re-muestrea `p_b`, `p_e`, `dp_b` y `dp_e` en la línea temporal del robot. |
| `training/nets.py` | `HistoryGruEncoder`; `DLOModel` con escalado de entradas, `bound_h0` y ruido de entrada solo en training; `build_model_from_checkpoint`. |
| `training/process_data.py` | Encoder con historia (`enc_history`), comandos (`enc_commands`), escalado del encoder (z-score) y del decoder ajustados en train, y `val_traj_ids` para validar con trayectorias enteras. |
| `training/train_simple.py` | Configuración nueva (ver abajo); early stopping que restaura el mejor modelo; AdamW. |
| `training/test.py`, `controllers/mpc*.py` | Cargan cualquier checkpoint (también los antiguos) con `build_model_from_checkpoint`; el MPC mantiene un buffer con la historia del encoder. |

Las pruebas del tracker se han hecho con datos sintéticos y paquetes UDP reales en localhost. Lo que cubren:
- **Escenarios:** el extremo pasa por encima de la base el 48% del tiempo, orden aleatorio en la lista, oclusiones de hasta 0.55 s, reflejos junto a cada marcador, marcadores fantasma y 1 de cada 4 frames.
- **Resultado:** **0 intercambios**, frente a 1746 frames intercambiados con la asignación antigua por Z.
- **Vaciado del socket:** con 200 paquetes acumulados procesa solo el último.
- **Pendiente:** no se ha probado con el OptiTrack real.

## 5. Acciones recomendadas (por orden de impacto)

1. **Volver a grabar el dataset con el tracker nuevo** y revisar `mocap_age` en los CSV. Debe quedarse por debajo de ~10 ms y sin saltos. Es lo que más mejora el modelo, y además lo que verá el MPC en tiempo real: el alineado offline no existe en el robot.
2. **Mientras tanto, entrenar con `ALIGN_MOCAP = True`** sobre los datos actuales (ya activado en `train_simple.py`).
3. **Usar el encoder GRU con historia** (`ENC_HISTORY = 10`, ya activado) con regularización suave: `DROPOUT = 0.05`, `NOISE_STD = 0.02`, `WEIGHT_DECAY = 1e-5`.
4. **Validar con trayectorias enteras** (`VAL_TRAJ_IDS`), no solo con el 20% final. El 80/20 temporal mide algo muy distinto de lo que necesita el MPC.
5. **Más trayectorias y más variadas.** La dispersión entre particiones (2.4–4.0 cm con la misma configuración) indica que el error depende mucho de qué trayectorias se validan.
6. **Arreglar los ficheros con bytes nulos** en el repo.
7. **Opcional:** bajar `window_size` de la media del tracker (7 frames ≈ 17 ms de retraso) o quitar el EMA del logger, si se quiere menos latencia. Son retrasos fijos que se pueden aprender, así que no es urgente.
