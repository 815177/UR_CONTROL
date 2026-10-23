from NatNetClient import NatNetClient
import time

# ======== CALLBACK: SE LLAMA POR CADA FRAME ========
def on_frame(frameData):
    mocap_data = frameData["mocap_data"]
    # print( mocap_data.rigid_body_data.rigid_body_list[0].get_as_string())
    # print(mocap_data.rigid_body_data.rigid_body_list[0].rb_marker_list[0].)
    # for i in range(mocap_data.labeled_marker_data.get_labeled_marker_count()):
    #     labeled_marker = mocap_data.labeled_marker_data.labeled_marker_list[i]
    #     print(labeled_marker.pos)
    labeled_marker = mocap_data.labeled_marker_data.labeled_marker_list[0]
    # print(labeled_marker.pos)

# ======== MAIN ========
if __name__ == "__main__":
    client = NatNetClient()

    # 🔧 CONFIGURA AQUÍ TUS IPs REALES
    client.set_client_address("169.254.126.181")    
    client.set_server_address("169.254.126.180")     
    client.set_use_multicast(False)        
    client.set_print_level(1)          

    client.new_frame_with_data_listener = on_frame

    print("Connecting to OptiTrack...")
    ok = client.run("d")
    if not ok:
        print("ERROR: no se pudo iniciar cliente NatNet")
        exit(1)

    print("Connected. Listening for marker data...\n")

   
