import socket
import threading
import datetime
import time
import ntplib
import pandas as pd
# ts = time.gmtime()
# recv_time =time.strftime("%X ", ts)
# print("current timestamp =" , ts)
# print("current time =",recv_time )
    
port = 9999
# server = "192.168.0.181"
server = socket.gethostbyname(socket.gethostname())
print(server)

# print(socket.gethostname())
FORMAT = 'utf-8'
msgno =0
ADDR = (server,port)
latency_l=[]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(ADDR)

print("[STARTING] SERVER is staring ...")


s.listen(100)


print("waiting for connections...")
while True:
    conn, addr = s.accept()
    while (msgno<100):

        name = conn.recv(1024).decode()
        
        msgno = msgno+1
        # print("connected with ", addr, name)
            
        conn.send(bytes("welcome !!","utf-8"))   
        
        name = float(name)

        # time_start = datetime.datetime.now().timestamp()
        # time_start = time.time()
        # # # time_start = datetime.datetime.utcnow().timestamp()
        # latency  = float(abs(time_start - name)*1000)

        ntp_client = ntplib.NTPClient()
        response = ntp_client.request('pool.ntp.org')
        latency  = float(abs(response.tx_time - float(name))*1000)
        
        # print("LATENCY for message ", msgno,  "= ", latency,"ms \n")
        print(latency)
        latency_l.append(latency)
df = pd.Dataframe(latency_l)
print(latency_l)
print("saving csv")
# 'F:\ASSIGNMENTS_QUIZ_TO_BE_SUBMITTED\SEM 2\WN\ASSIGNMENT1\wireless.csv'
df.to_csv('wireless.csv')
conn.close()
        

s.close()    





