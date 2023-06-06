# -*- coding: utf-8 -*-
"""P1Discrete and continuous membership functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_ZW7_hPkci7auUpAW7EvQkdel-s3lq7O
"""

import matplotlib.pyplot as plt
import numpy as np


def triangle(ini,fin,a,b,c,step):  #--------------------------TRIANGLE-----------------------
  u=np.arange(ini,fin,step)
  t=[]

  for k in u:
    if (-ini<=k<a):
      t.append(0)
    elif(a<=k<b):
      t.append(((k-b)/(b-a))+1)
    elif(b<=k<c):
      t.append(-(k-c)/(c-b))
    else:
      t.append(0)

  return ([u,t])

def trapezoid(ini,fin,a,b,c,d,step):   #--------------------------TRAPEZOID-----------------------
  u=np.arange(ini,fin,step)
  tr=[]

  for k in u:
    if (ini<=k<a):
      tr.append(0)
    elif(a<=k<b):
      tr.append(((k-b)/(b-a))+1)
    elif(b<=k<c):
      tr.append(1)
    elif(c<=k<=d):
      tr.append(-(k-d)/(d-c))
    else:
      tr.append(0)

  return ([u,tr])

def gaussian(ini,fin,c,sig,step): #--------------------------GAUSSIAN-----------------------
  u=np.arange(ini,fin,step)
  gs=[]

  for k in u:
    gs.append(np.exp(-(1/2)*((k-c)/sig)**2))
  return([u,gs])

def bell(ini,fin,a,b,c,step):  #--------------------------BELL-----------------------
  u=np.arange(ini,fin,step)
  bll=[]

  for k in u:
    bll.append(1/(1+(np.abs((k-c)/a))**(2*b)))

  return([u,bll])

def sigmoid(ini,fin,a,c,step):  #--------------------------SIGMOID-----------------------
  u=np.arange(ini,fin,step)
  sgm=[]

  for k in u:
    sgm.append(1/(1+np.exp(-a*(k-c))))

  return([u,sgm])

  
#---------------------------------- BUILDING GRAPHS------------------------


universe1,triangle=triangle(0,200,50,100,150,0.5)
universe2,trapezoid1=trapezoid(0,200,0,0,45,80,0.5)
universe3,trapezoid2=trapezoid(0,200,120,155,200,200,0.5)

plt.figure(1)
plt.plot(universe1,triangle,universe2,trapezoid1,universe3,trapezoid2)
plt.legend(["Triangle", "Trapezoid A", "Trapezoid B"],loc="best")
plt.title("Discrete Functions")
plt.ylabel("Membership Degree")
plt.xlabel("Universe")
plt.grid(True)

universe4,gauss=gaussian(0,200,100,25,0.5)
universe5,bell=bell(0,200,40,10,0,0.5)
universe6,sigmoid=sigmoid(0,200,0.1,140,0.5)

plt.figure(2)
plt.plot(universe4,gauss,universe5,bell,universe6,sigmoid)
plt.legend(["Gaussian","Bell","Sigmoid"])
plt.title("Continuous Functions")
plt.ylabel("Membership Degree")
plt.xlabel("Universe")
plt.grid(True)