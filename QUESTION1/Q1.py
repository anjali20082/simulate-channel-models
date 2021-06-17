#QUESTION1

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import erfc #erfc/Q function
import math 
import cmath 
import time

N = 10**5 # Number of symbols to transmit
SNRdBs = np.arange(start=-4,stop = 13, step = 2) # Eb/N0 range in dB for simulation

Inp = np.random.randint(0, 2, (1, N))
yy =np.zeros((1, int(N/4)))
recv_yy =np.zeros((1, int(N/4)))
yy_16QAM =[]

yy_32 =np.zeros((1, int(N/5)))
recv_yy32 = np.zeros((1, int(N/5)))
yy_32QAM = []

i=0
j=0

BER_16QAM = np.zeros((1,len(SNRdBs)))
BER_32QAM = np.zeros((1, len(SNRdBs)))

count = 0





BER_theory_bpsk = 0.5*erfc(np.sqrt(10**(SNRdBs/10)))
BER_theory_qpsk = 0.5*(erfc(np.sqrt(10**(SNRdBs/10)))-0.25*erfc(np.sqrt(10**(SNRdBs/10)))**2)

BER_theory_4QAM = (1/math.log2(4))*(2*(1-np.sqrt(1/4))*erfc(np.sqrt((3*math.log2(4)*(10**(SNRdBs/10)))/(2*(4-1)))))

BER_theory_16QAM = (1/math.log2(16))*(2*(1-np.sqrt(1/16))*erfc(np.sqrt((3*math.log2(16)*(10**(SNRdBs/10)))/(2*(16-1)))))

BER_theory_32QAM = (1/math.log2(32))*(2*(1-np.sqrt(1/32))*erfc(np.sqrt((3*math.log2(32)*(10**(SNRdBs/10)))/(2*(32-1)))))




#NOise
def AWGN_noise(EbNo, N, sym_num, bit_num):

    
    # Modulation level 16QAM: 4, 32QAM: 5
    QAM_order = int(bit_num / sym_num)
    SNR = QAM_order * 10 ** (EbNo / 10)
    No = 1 / SNR
    noise = np.random.normal(0, np.sqrt(No / 2), (N, sym_num))+ 1j * np.random.normal(0, np.sqrt(No / 2), (N, sym_num))
    return noise

#QAM_DEMODULATION
def QAM16_to_bit(rc_symbol):
    temp = rc_symbol.shape
    # print("received signal shape", temp, temp[0], temp[1])
    N = temp[0]
    sym_num = temp[1]
    bit_num = sym_num * 4

    rc_symbol = rc_symbol * np.sqrt(10)    
    rc_symbol_real = np.real(rc_symbol)
    rc_symbol_imag = np.imag(rc_symbol)
    # print("Received signal",RX_symbol)
    # print("RX_symbol_real before",RX_symbol_real)
    # print("RX_symbol_imag before",RX_symbol_imag)

    real_copy = rc_symbol_real.copy()
    imag_copy = rc_symbol_imag.copy()

    rc_symbol_real[real_copy > 2] = 3
    rc_symbol_real[2 > real_copy] = 1
    rc_symbol_real[0 > real_copy] = -1
    rc_symbol_real[-2 > real_copy] = -3

    rc_symbol_imag[imag_copy > 2] = 3
    rc_symbol_imag[2 > imag_copy] = 1
    rc_symbol_imag[0 > imag_copy] = -1
    rc_symbol_imag[-2 > imag_copy] = -3

    # print("RX_symbol_real after", RX_symbol_real)
    # print("RX_symbol_imag after",RX_symbol_imag)
    # print("final RX_symbol", RX_symbol)

    temp = np.zeros((N, sym_num * 2))
    # print("temp before", temp)
    temp[:, 0:sym_num*2:2] = rc_symbol_real
    temp[:, 1:sym_num*2:2] = rc_symbol_imag
    # print("temp after", temp)

    temp2 = np.zeros((N, sym_num * 2)) + 1j * np.zeros((N, sym_num * 2))
    # print("temp2 before", temp2)
    temp2[temp == 3] = 1 - 0j
    temp2[temp == 1] = 1 + 1j
    temp2[temp == -1] = - 0 + 1j
    temp2[temp == -3] = - 0 - 0j
    # print("temp2 after", temp2)

    bits_rc = np.zeros((N, bit_num))
    # print("received in bits before", RX_bit)
    bits_rc[:, 0:bit_num:2] = np.real(temp2)
    bits_rc[:, 1:bit_num:2] = np.imag(temp2)
    # print("received in bits after", RX_bit)
    return bits_rc


def QAM32_to_bit(rc_symbol):

    

    temp = rc_symbol.shape
    # print("received signal shape", temp)
    
    p=0
    i=0
    N = temp[0]
    sym_num = temp[1]
    bit_num = sym_num * 5
    Inp =np.zeros((N, bit_num))

    rc_symbol = rc_symbol * np.sqrt(20)    
    rc_symbol_real = np.real(rc_symbol)
    rc_symbol_imag = np.imag(rc_symbol)

    real_copy = rc_symbol_real.copy()
    imag_copy = rc_symbol_imag.copy()


    rc_symbol_real[real_copy > 4] = 5
    rc_symbol_real[real_copy > 2] = 3
    rc_symbol_real[2 > real_copy] = 1
    rc_symbol_real[0 > real_copy] = -1
    rc_symbol_real[-2 > real_copy] = -3
    rc_symbol_real[-4 > real_copy] = -5


    rc_symbol_imag[imag_copy > 4] = 5
    rc_symbol_imag[imag_copy > 2] = 3
    rc_symbol_imag[2 > imag_copy] = 1
    rc_symbol_imag[0 > imag_copy] = -1
    rc_symbol_imag[-2 > imag_copy] = -3
    rc_symbol_imag[-4 > imag_copy] = -5

    
    while(p < sym_num):
        if( rc_symbol[0][p] == (complex(5.0,1.0))) :

            Inp[0][i] = 1
            Inp[0][i+1]= 1
            Inp[0][i+2]= 1
            Inp[0][i+3]= 1
            Inp[0][i+4]= 0

        
        elif( rc_symbol[0][p] == (complex(3.0,1.0))) :
           
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(1.0,1.0))) :
            
            Inp[0][i]=1 
            Inp[0][i+1]= 0
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0
            
        elif(rc_symbol[0][p] == (complex(5.0,3.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0



        elif( rc_symbol[0][p] == (complex(3.0,3.0))) :
           
            Inp[0][i]= 0 
            Inp[0][i+1]= 0
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(1.0,3.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(3.0,5.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0
            
        elif( rc_symbol[0][p] == (complex(1.0,5.0))) :
           
            Inp[0][i]=0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0




        elif(rc_symbol[0][p] == (complex(-1.0,1.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(-3.0,1.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(-5.0,1.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 0
            
        elif( rc_symbol[0][p] == (complex(-1.0,3.0))) :
           
            Inp[0][i]= 0 
            Inp[0][i+1]= 1
            Inp[0][i+2]=1 
            Inp[0][i+3]=0 
            Inp[0][i+4]= 0



        elif(rc_symbol[0][p] == (complex(-3.0,3.0))) :
            
            Inp[0][i]=1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(-5.0,3.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0

        elif(rc_symbol[0][p] == (complex(-1.0,5.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0
            Inp[0][i+4]= 0
            
        elif( rc_symbol[0][p] == (complex(-3.0,5.0))) :
           
            Inp[0][i]= 0 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 0




        elif(rc_symbol[0][p] == (complex(-1.0,-1.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(-3.0,-1.0))) :
            
            Inp[0][i]= 1
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(-5.0,-1.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0  
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

            
        elif(rc_symbol[0][p] == (complex(-1.0,-3.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1



        
        elif(rc_symbol[0][p] == (complex(-3.0,-3.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1


        elif(rc_symbol[0][p] == (complex(-5.0,-3.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(-1.0,-5.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1
            
        elif(rc_symbol[0][p] == (complex(-3.0,-5.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1




        elif(rc_symbol[0][p] ==(complex(1.0,-1.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(3.0,-1.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(5.0,-1.0))) :
            
            Inp[0][i]=1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 0 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(1.0,-3.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 0 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1




        elif(rc_symbol[0][p] == (complex(3.0,-3.0))) :
            
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1

        elif(rc_symbol[0][p] == (complex(5.0,-3.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 0 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1

        elif( rc_symbol[0][p] == (complex(1.0,-5.0))) :
            Inp[0][i]= 0 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1 
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1

            
        elif(rc_symbol[0][p] ==(complex(3.0,-5.0))) :
            
            Inp[0][i]= 1 
            Inp[0][i+1]= 1 
            Inp[0][i+2]= 1
            Inp[0][i+3]= 1 
            Inp[0][i+4]= 1
        
        i += 5
        p+=1

    return Inp


# Generating 16QAM transmission signal with complex values
while(j<len(Inp[0])):
# while(i<N):   
  # if(len(Inp) < N):
#   print(i, a, b)

  if((Inp[0][j]== 0) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 0)) :
    # y = np.sqrt(1/10)*(-3+3j)
    y = np.sqrt(1/10)*(complex(-3.0,3.0))
  
  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 0)):
    y = np.sqrt(1/10)*(complex(-1.0,3.0))

  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 1)):
    y = np.sqrt(1/10)*(complex(-3.0,1.0))
    
  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 1)) :
    y = np.sqrt(1/10)*(complex(-1.0,1.0))




  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 0)) :
    y = np.sqrt(1/10)*(complex(1.0,3.0))

  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 0)):
    y = np.sqrt(1/10)*(complex(3.0,3.0))

  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 1)):
    y = np.sqrt(1/10)*(complex(1.0,1.0))
    
  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 1) & (Inp[0][j+3]== 1)) :
    y = np.sqrt(1/10)*(complex(3.0,1.0))



  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 1)) :
    y = np.sqrt(1/10)*(complex(-3.0,-1.0))

  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 1)):
    y = np.sqrt(1/10)*(complex(-1.0,-1.0))

  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 0)):
    y = np.sqrt(1/10)*(complex(-3.0,-3.0))
    
  elif((Inp[0][j]== 0) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 0)) :
    y = np.sqrt(1/10)*(complex(-1.0,-3.0))





  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 1)) :
    y = np.sqrt(1/10)*(complex(1.0,-1.0))

  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 1)):
    y = np.sqrt(1/10)*(complex(3.0,-1.0))

  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 1) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 0)):
    y = np.sqrt(1/10)*(complex(1.0,-3.0))
    
  elif((Inp[0][j]== 1) & (Inp[0][j+1]== 0) & (Inp[0][j+2]== 0) & (Inp[0][j+3]== 0)) :
    y = np.sqrt(1/10)*(complex(3.0,-3.0))

  yy_16QAM.append(y)
  # np.append(yy[0], np.ravel(y))
  j+=4


yy =np.array(yy_16QAM)
yy = np.reshape(yy,(1,int(N/4)))
# print(yy)




# Generating 32QAM transmission signal with complex values
while(i<len(Inp[0])):

  if((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(5.0,1.0))
  
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(3.0,1.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(1.0,1.0))
    
  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(5.0,3.0))



  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(3.0,3.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(1.0,3.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(3.0,5.0))
    
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(1.0,5.0))




  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-1.0,1.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-3.0,1.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-5.0,1.0))
    
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-1.0,3.0))





  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-3.0,3.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-5.0,3.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-1.0,5.0))
    
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 0)) :
    y = np.sqrt(1/20)*(complex(-3.0,5.0))




  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-1.0,-1.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-3.0,-1.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-5.0,-1.0))
    
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-1.0,-3.0))



 
  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-3.0,-3.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-5.0,-3.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-1.0,-5.0))
    
  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(-3.0,-5.0))




  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(1.0,-1.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(3.0,-1.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 0) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(5.0,-1.0))
    
  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 0) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(1.0,-3.0))




  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(3.0,-3.0))

  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 0) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(5.0,-3.0))

  elif((Inp[0][i]== 0) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(1.0,-5.0))
    
  elif((Inp[0][i]== 1) & (Inp[0][i+1]== 1) & (Inp[0][i+2]== 1) & (Inp[0][i+3]== 1) & (Inp[0][i+4]== 1)) :
    y = np.sqrt(1/20)*(complex(3.0,-5.0))



  yy_32QAM.append(y)
  # np.append(yy[0], np.ravel(y))
  i+=5


yy_32 =np.array(yy_32QAM)
yy_32 = np.reshape(yy_32,(1,int(N/5)))
# print(yy_32.shape)


for snrdb in(SNRdBs):
  noise = AWGN_noise(snrdb, 1, len(yy[0, :]), N)
  recv_yy = yy+noise

  noise_32 = AWGN_noise(snrdb, 1, len(yy_32[0, :]), N)
  recv_yy32 = yy_32 + noise_32
  # print(recv_yy)
  # error = recv_yy - yy
  # print(error)
  
  RX_bit_16QAM = QAM16_to_bit(recv_yy)
  rx_bit_32QAM = QAM32_to_bit(recv_yy32)
#   print("Received 16 qam symbols to bits",RX_bit_16QAM.shape)
  # BER calculation
  error_sum_16QAM = np.sum(np.abs(Inp - RX_bit_16QAM))
  error_sum_32QAM = np.sum(np.abs(Inp - rx_bit_32QAM))
#   print("Error in  16 qam symbols",error_sum_16QAM)
  BER_16QAM[0, count] = error_sum_16QAM / (1 * N)
  # print(BER_16QAM)
  BER_32QAM[0, count] = error_sum_32QAM/(1*N)
  # print(BER_32QAM)

  count+=1
 

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(SNRdBs,BER_theory_bpsk, color ='r', marker='X',linestyle='-',label='BPSK Theory')
ax.semilogy(SNRdBs,BER_theory_qpsk,color ='y',marker='o',linestyle='-',label='QPSK Theory')
ax.semilogy(SNRdBs,BER_theory_4QAM,color ='g',marker='',linestyle='-',label='4QAM Theory')
ax.semilogy(SNRdBs,BER_theory_16QAM,color ='cyan',marker='',linestyle='-',label='16QAM Theory')
# ax.semilogy(SNRdBs,BER_theory_32QAM,color ='b',marker='',linestyle='-',label='32QAM Theory')

plt.semilogy(SNRdBs, BER_16QAM[0,:],color ='r' ,marker ='o',label='16QAM Simulated')
plt.semilogy(SNRdBs, BER_32QAM[0,:],color ='orange' ,marker ='o',label='32QAM Simulated')
ax.set_xlabel('$E_b/N_0(dB)$');ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error for various modulation schemes over AWGN channel')
ax.set_xlim(-5,13)
# ax.set_xlim(0,20)
# plt.ylim(10**(-5), 10**(-0))
# ax = plt.gca()
ax.set_yscale('log')
ax.grid(True);
ax.legend();
plt.show()