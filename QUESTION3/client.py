import socket
from time import ctime
import datetime
import time
import ntplib

PORT = 9999
FORMAT = 'utf-8'
SERVER = "192.168.0.181" # wifi
# SERVER = "192.168.43.233" # wireless mobile
# SERVER = "192.168.42.62"  # wired-tether
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

# print(response.tx_time)

for i in range(100):
    time.sleep(2)
    # client.connect(ADDR)
    
    # time_start = datetime.datetime.now().timestamp()
    # time_start = time.time()
    # time_start = datetime.datetime.utcnow().timestamp()
    # time_start = str(time_start)
    # client.send(bytes(time_start,"utf-8"))

    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('pool.ntp.org')
    curr_time = response.tx_time
    curr_time= str(curr_time)
    client.send(bytes(curr_time,"utf-8"))

    print(client.recv(1024).decode())



