import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 1511))

print("Esperando paquetes NatNet...")

while True:
    data, addr = sock.recvfrom(65535)
    print("Paquete recibido desde:", addr, "size:", len(data))
