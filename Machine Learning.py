import time
import threading
import spidev
import csv
import math
import RPi.GPIO as GPIO # To use GPIO pins
import serial
import pandas as pd
import numpy as np
from numpy import mean, sqrt, square, arange
from numpy import set_printoptions
from numpy import interp
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
import Adafruit_PCA9685
import sys

spi=spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=1350000

ledPIN = 27
ledPIN1 = 17
ledPIN2 = 22
ledPIN3 = 19 #ch1
ledPIN4 = 26 #ch2

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

rangee=100
gerakan="FL"
emgch1=[]
emgch2=[]
emgexch1=[333]
emgexch2=[333]
emgc1=[]
emgc2=[]
Time=0.1
Timeex=1
Timeml=1
delay=0.001
delayex=0.1
delayml=0.1

mavch1= 0
ssich1= 0
varch1= 0
rmsch1= 0
mavch2= 0
ssich2= 0
varch2= 0
rmsch2= 0
gn= ""
knn= ""
vm= ""
dt= ""

def ReadChannel(channel):
    adc=spi.xfer([1,(8+channel)<<4,0])
    data=((adc[1]&3)<<8)+adc[2]
    return data

def servo():
    global knn
    global ledPIN
    global ledPIN1
    global ledPIN2
    ml=""
    while True:
        while (ml!=knn):
            if(knn=="HC"):
                pwm.set_pwm(1, 0, 200)
                pwm.set_pwm(2, 0, 550)
                pwm.set_pwm(0, 0, 350)
                GPIO.output(ledPIN,GPIO.HIGH)
                GPIO.output(ledPIN1,GPIO.LOW)
                GPIO.output(ledPIN2,GPIO.LOW)
                ml=knn
            elif(knn=="FL"):
                pwm.set_pwm(1, 0, 550)
                pwm.set_pwm(2, 0, 200)
                pwm.set_pwm(0, 0, 500)
                GPIO.output(ledPIN,GPIO.LOW)
                GPIO.output(ledPIN1,GPIO.HIGH)
                GPIO.output(ledPIN2,GPIO.LOW)
                #print("ok")
                ml=knn
            elif(knn=="EX"):
                pwm.set_pwm(1, 0, 550)
                pwm.set_pwm(2, 0, 200)
                pwm.set_pwm(0, 0, 250)
                GPIO.output(ledPIN,GPIO.LOW)
                GPIO.output(ledPIN1,GPIO.LOW)
                GPIO.output(ledPIN2,GPIO.HIGH)
                #print("ok")
                ml=knn
            else:
                pwm.set_pwm(1, 0, 550)
                pwm.set_pwm(2, 0, 200)
                pwm.set_pwm(0, 0, 350)
                GPIO.output(ledPIN,GPIO.LOW)
                GPIO.output(ledPIN1,GPIO.LOW)
                GPIO.output(ledPIN2,GPIO.LOW)
                ml=knn
                #print("ok")
        time.sleep(0.1)
def machine_learning():
    global mavch1
    global ssich1
    global varch1
    global rmsch1
    global mavch2
    global ssich2
    global varch2
    global rmsch2
    global gn
    global knn
    global svm
    global dt

    global Timeml
    timer_a=time.time()
    global delayml
    sampling=Timeml/delayml
    
    #from sklearn.naive_bayes import GaussianNB
    try:
        dataset = pd.read_csv(r"/home/pi/DataSet Gabungan.csv")
        datatest = pd.read_csv(r"/home/pi/DataTes Gabungan.csv")
        
        X_train = dataset[['mavch1','mavch2']].values
        Y_train = dataset["label"].values

        k = 9
        kNN = neighbors.KNeighborsClassifier(n_neighbors = k, weights='distance')
        kNN.fit(X_train, Y_train)

        #gnb = GaussianNB()
        #gnb.fit(X_train, Y_train)

        #DT = tree.DecisionTreeClassifier(max_depth=3)
        #DT = DT.fit(X_train, Y_train)

        #dSVM = svm.SVC(gamma=0.2, C=0.9, kernel='rbf') # one versus one SVM
        #dSVM.fit(X_train, Y_train)
    except:
        print("error read dataset/csv")

    while True:
        start = time.time()
        ml=0
        while ml<sampling:
            time.sleep(delayml)
            knn = kNN.predict([[mavch1,mavch2]])
            #gn = gnb.predict([[mavch1,ssich1,varch1,rmsch1,mavch2,ssich2,varch2,rmsch2]])
            #dt = DT.predict([[mavch1,ssich1,varch1,rmsch1,mavch2,ssich2,varch2,rmsch2]])
            #svm = dSVM.predict([[mavch1,ssich1,varch1,rmsch1,mavch2,ssich2,varch2,rmsch2]])

            print('k-NN:', knn)
            #print('Naive Bayes:', gn)
            #print('DT:', dt)
            #print('SVM:', svm)
            ml +=10
        end = time.time()
        adjust3 = (((end-start)-(Time*1))/ml)
        #print(end-start,ml,delayml,adjust3)
        if (delayml-adjust3>0):
            delayml -= adjust3
        
def extraction():
    global emgch1
    global emgch2
    global emgexch1
    global emgexch2
    global emgc1
    global emgc2
    global rangee

    global mavch1
    global ssich1
    global varch1
    global rmsch1
    global mavch2
    global ssich2
    global varch2
    global rmsch2
    global gn
    global knn
    global svm
    global dt

    global ledPIN3
    global ledPIN4
    
    global Timeex
    timer_a=time.time()
    global delayex
    sampling=Timeex/delayex
   
    while True:
        start = time.time()
        ex=0
        while ex<sampling:
            time.sleep(delayex)
            arrch1= pd.DataFrame(emgexch1)
            arrch2= pd.DataFrame(emgexch2)
            a= np.mean(np.absolute(arrch1))#mav
            #b= np.sum(arrch1**2)#SSI
            #c= np.sum(arrch1**2)/(rangee-1)#var
            #d= np.sqrt(np.sum(arrch1**2)/(rangee))#rms
            e= np.mean(np.absolute(arrch2))#mav
            #f= np.sum(arrch2**2)#SSI
            #g= np.sum(arrch2**2)/(rangee-1)#var
            #h= np.sqrt(np.sum(arrch2**2)/(rangee))#rms
            mavch1 = a.iloc[0]
            #ssich1 = b.iloc[0]
            #varch1 = c.iloc[0]
            #rmsch1 = d.iloc[0]
            mavch2 = e.iloc[0]
            #ssich2 = f.iloc[0]
            #varch2 = g.iloc[0]
            #rmsch2 = h.iloc[0]
            #print(mavch1)
            led1=interp(mavch1,[0,120],[0,100])
            led2=interp(mavch2,[0,120],[0,100])
            ledch1.ChangeDutyCycle(led1)
            ledch2.ChangeDutyCycle(led2)
            ex +=10
            
        end = time.time()
        adjust2 = (((end-start)-(Time*1))/ex)
        #print(end-start,ex,delayex,adjust2)
        if (delayex-adjust2>0):
            delayex -= adjust2
            
def get_data():

    global emgch1
    global emgch2
    global emgexch1
    global emgexch2
    global delay
    global Time
    sampling=Time/delay
    adjust1=0

    #variabel filter

    b0 =0.78429785289303577
    b1 =-4.7057871173582146
    b2 =11.764467793395536
    b3 =-15.685957057860715
    b4 =11.764467793395536
    b5 =-4.7057871173582146
    b6 =0.78429785289303577
    a0 =1
    a1 =-5.5145351211661646
    a2 =12.689113056515138
    a3 =-15.593635210704097
    a4 =10.793296670485377
    a5 =-3.9893594042308824
    a6 =0.6151231220526282

    y1ch1=0
    y2ch1=0
    y3ch1=0
    y4ch1=0
    y5ch1=0
    y6ch1=0
    yach1=0
    xach1=0
    x0ch1=0
    x1ch1=0
    x2ch1=0
    x3ch1=0
    x4ch1=0
    x5ch1=0
    x6ch1=0

    y1ch2=0
    y2ch2=0
    y3ch2=0
    y4ch2=0
    y5ch2=0
    y6ch2=0
    yach2=0
    xach2=0
    x0ch2=0
    x1ch2=0
    x2ch2=0
    x3ch2=0
    x4ch2=0
    x5ch2=0
    x6ch2=0
    
    while True:
        start = time.time()
        emgch1=[]
        emgch2=[]
        while len(emgch1)<sampling and len(emgch2)<sampling:
            time.sleep(delay)
            ch1= ReadChannel(0)
            ch2= ReadChannel(1)
            y6ch1=y5ch1
            y5ch1=y4ch1
            y4ch1=y3ch1
            y3ch1=y2ch1
            y2ch1=y1ch1
            y1ch1=yach1
            x6ch1=x5ch1
            x5ch1=x4ch1
            x4ch1=x3ch1
            x3ch1=x2ch1
            x2ch1=x1ch1
            x1ch1=x0ch1
            x0ch1=xach1
            xach1=ch1
            
            y6ch2=y5ch2
            y5ch2=y4ch2
            y4ch2=y3ch2
            y3ch2=y2ch2
            y2ch2=y1ch2
            y1ch2=yach2
            x6ch2=x5ch2
            x5ch2=x4ch2
            x4ch2=x3ch2
            x3ch2=x2ch2
            x2ch2=x1ch2
            x1ch2=x0ch2
            x0ch2=xach2
            xach2=ch2
            yach1 = b0*x0ch1 + b1*x1ch1 + b2*x2ch1 + b3*x3ch1 + b4*x4ch1 + b5*x5ch1 + b6*x6ch1 - a1*y1ch1 - a2*y2ch1 - a3*y3ch1 - a4*y4ch1 - a5*y5ch1 - a6*y6ch1
            yach2 = b0*x0ch2 + b1*x1ch2 + b2*x2ch2 + b3*x3ch2 + b4*x4ch2 + b5*x5ch2 + b6*x6ch2 - a1*y1ch2 - a2*y2ch2 - a3*y3ch2 - a4*y4ch2 - a5*y5ch2 - a6*y6ch2
            emgch1.append(yach1)
            emgch2.append(yach2)
            if len(emgch1)==sampling and len(emgch2)==sampling:
                emgexch1=emgch1
                emgexch2=emgch2
        end = time.time()
        adjust1 = (((end-start)-(Time*1))/len(emgch1)/1)
        #print(end-start,len(emgch1),delay,adjust1)
        if (delay-adjust1>0):
            delay -= adjust1

try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ledPIN, GPIO.OUT)
    GPIO.setup(ledPIN1, GPIO.OUT)
    GPIO.setup(ledPIN2, GPIO.OUT)
    GPIO.setup(ledPIN3, GPIO.OUT)
    GPIO.setup(ledPIN4, GPIO.OUT)
    ledch1=GPIO.PWM(ledPIN3,50)
    ledch2=GPIO.PWM(ledPIN4,50)
    #GPIO.cleanup()
    GPIO.output(ledPIN,GPIO.HIGH)
    GPIO.output(ledPIN1,GPIO.HIGH)
    GPIO.output(ledPIN2,GPIO.HIGH)
    ledch1.start(100)
    ledch2.start(100)
    pwm.set_pwm(1, 0, 550)
    pwm.set_pwm(2, 0, 200)
    pwm.set_pwm(0, 0, 350)
    time.sleep(2)
    #GPIO.cleanup()
except:
    print("Error Initialization Servo")

try:
    t1= threading.Thread(target=get_data,)
    t2= threading.Thread(target=extraction,)
    t3= threading.Thread(target=machine_learning,)
    t4= threading.Thread(target=servo,)
except:
    print("Error Thread")
    
try:
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
except:
    print("Error Start Thread")
