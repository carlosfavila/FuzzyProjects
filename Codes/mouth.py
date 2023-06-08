import matplotlib.pyplot as plt
import numpy as np

def hardlim(x):
    if (x<0).all():
        return 0
    else:
        return 1

#------------------------------------------------------------MOUTH------------------------------------------------------------
#Yl7 = -(1/4)*X + 1
#Yl8 = -(3/4)*X + 1
#Xl9 = 1

#So, the weight line equations are:
#Yw7 = 1/(1/4)*X 
#Yw8 = 1/(3/4)*X 
#Xw9 = 0

#calculating weights for mouth
#so we want the line of the weight 7 points inside the mouth
#proposing x=-1, with Yw7=1/(1/4)*X we get Yw7 = -4

#and we also want the line of the weight 8 points at the middle of the thre lines
#proposing x=1, with Yw8=1/(3/4)*X we get Yw8 = 4/3

#and we also want the line of the weight 9 points inside the mouth
#in this case, the weight 9 is a vertical line, so we can propose x=-1 and in Yw9=0

w17=[-1,-4]
w18=[1,4/3]
w19=[-1,0]


#calculating bias for mouth
# a point on line 7 is [0,1]
b17=-np.dot(w17,[0,1])

# a point on line 8 is [0,0.75]
b18=-np.dot(w18,[0,1])

# a point on line 9 is [0,-1]
b19=-np.dot(w19,[1,0.5])

W1=[w17,w18,w19]
B1=[[b17],[b18],[b19]]

W2=[1,1,1]
B2=-2.5

xp=np.linspace(0,8)
yp=np.linspace(0,1.5)
for x in xp:
    for y in yp:
        a1=np.dot(W1,[[x],[y]])+B1
        for i in range(len(a1)):
            a1[i]=hardlim(a1[i])
        a2=hardlim(np.dot(W2,a1)+B2)
        if a2==1:
            plt.plot(x,y,'bo')
        else:
            plt.plot(x,y,'ro')

plt.figure(1)


xp=np.linspace(0,1.5)
yp=np.linspace(0,1.5)
for x in xp:
    for y in yp:
        a1=np.dot(W1,[[x],[y]])+B1
        for i in range(len(a1)):
            a1[i]=hardlim(a1[i])
        a2=hardlim(np.dot(W2,a1)+B2)
        if a2==1:
            plt.plot(x,y,'bo')
        else:
            plt.plot(x,y,'ro')


plt.grid(True)

plt.show()

#280601
