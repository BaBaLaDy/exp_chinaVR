# -*- coding: utf-8 -*-

"""
Created on Wed Jan 27 18:10:21 2021

@author: nickxia
"""

from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
# import pywt
from pyemd import EMD



def np_move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))

class Get_Feature():
       
    def __init__ (self,data,code):
        self.data = np_move_avg(data,15,mode="same")
        self.code = code  
        self.result = []  
##############Time_Domain################    
    def mean(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        self.result.append(np.mean(tem))
    
    def var(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        self.result.append(np.var(tem))
    
    def standard(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        self.result.append(np.std(tem))
    
    def Per_75(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        self.result.append(np.percentile(tem,75))
    
    def Per_25(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        sfper = np.percentile(tem,75)
        Q1 = np.percentile(tem,25)
        self.result.append((sfper - Q1))
#################Frequency_Domain###################
####Estimate power spectral density using Welch’s method####.
    def mean_PSD(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        freqs, psd = signal.welch(tem)
        self.result.append(np.mean(psd))
    
    def med_PSD(self):
       #for i in range (self.data.shape[1]):
        tem = self.data
        freqs, psd = signal.welch(tem)
        self.result.append(np.median(psd))
    
    def MNF_PSO(self):
        #mean frequecy of PSD
        #for i in range (self.data.shape[1]):
        tem = self.data
        freqs, psd = signal.welch(tem)
        #print(len(freqs),len(psd))
        s = []
        for j in range (len(freqs)):
            s.append(freqs[j]*psd[j])
        self.result.append((sum(s)/sum(psd)))
            
    def MDF_PSO(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        freqs, psd = signal.welch(tem)
        s = 0
        for j in range (len(freqs)):
            s = s + psd[j]
            if (sum(psd)/2)<= s:
                MDF = freqs[j]
        self.result.append(MDF)

    def Entropy(self):
        #for i in range (self.data.shape[1]):
        tem = self.data
        _, psd = signal.welch(tem,100)
        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        self.result.append(se)
        
    def EMD(self):
        #for i in range (self.data.shape[1]):
        
        tem = self.data        
        imf = []
        emd = EMD
        # IMFs = emd.emd(tem)
        IMFs = emd.EMD(tem)
        for n, j in enumerate (IMFs):
            if n<=2: # pick the previous 3 item of IMF
                #imf.append(i)
                e = (sum(j**2))/5
                imf.append(e)
                res = tem-j
                res_e = (sum(res**2))/5
                imf.append(res_e)
        self.result.append(imf[0])
        self.result.append(imf[1])
        self.result.append(imf[2])
        self.result.append(imf[3])

    def FFT(self):
        #for i in range (self.data.shape[1]):
        tem = self.data 
        N = 64
        cor_X= fft(tem)
        ps_cor = np.abs(cor_X)
        self.result.append(ps_cor[1])
        self.result.append(ps_cor[2])
        self.result.append(ps_cor[3])
        self.result.append(ps_cor[4])
        self.result.append(ps_cor[5])
        self.result.append(ps_cor[6])
        self.result.append(ps_cor[7])
        self.result.append(ps_cor[8])
        self.result.append(ps_cor[9])
        self.result.append(ps_cor[10])


###################Time-Frequency Domain########################
    def cal_result(self):
        #print("calculating the handcrafted features")
        if self.code[0] == 1:
            self.mean()
        if self.code[1] == 1:
            self.var()
        if self.code[2] == 1:
            self.standard()
        if self.code[3] == 1:
            self.Per_75()
        if self.code[4] == 1:
            self.Per_25() 
        if self.code[5] == 1:
            self.mean_PSD()
        if self.code[6] == 1:
            self.med_PSD()
        if self.code[7] == 1:
            self.MNF_PSO()  
        if self.code[8] == 1:
            self.MDF_PSO()  
        if self.code[9] == 1:
            self.Entropy()  
        if self.code[10] == 1:
            self.EMD()
        if self.code[11] == 1:
            self.FFT()
        return self.result
    
    
        
        