import matplotlib.pyplot as plt
import numpy as np

def hardlim(x):
    if (x<0).all():
        return 0
    else:
        return 1


#------------------------------------------------------------MOUTH------------------------------------------------------------
#first neuron for the mouth
#Yl7 = -(1/4)x+1

#second neuron for the mouth
#Yl8 = 0.75 from x=1 to x=7

#third neuron for the mouth
#Yl9 = (1/4)x-1

#fourth neuron for the mouth
#Yl10 = -(3/4)x+1

#fifth neuron for the mouth
#Yl11 = 0.25 from x=1 to x=7

#sixth neuron for the mouth
#Yl12 = (3/4)x-5

#Calculating the weights for the mouth
#So, the weight line equations are:
# Yw17 = (1/4)X
# Yw18 = inf
# Yw19 = -(1/4)X
# Yw110 = (3/4)X
# Yw111 = inf
# Yw112 = -(3/4)X

# so, we want the line of the weight 7 points inside the mouth
# proposing x = -1, with Yw17=(1/4)x we get Yw17 = -1/4

# and we also want the line of the weight 8 points inside the mouth
# but this one is a vertical line so, we can propose x=0 and in Yw18=-1

# and we also want the line of the weight 9 points inside the mouth
# proposing x = 1, with Yw19=-(1/4)x we get Yw19 = -1/4

# and we also want the line of the weight 10 points inside the mouth
# proposing x = 1, with Yw110=(3/4)x we get Yw110 = 3/4

# and we also want the line of the weight 11 points inside the mouth
# but this one is a vertical line so, we can propose x=0 and in Yw111=1

# and we also want the line of the weight 12 points inside the mouth
# proposing x = -1, with Yw112=-(3/4)x we get Yw112 = 3/4

w17=[-1,-(1/4)]
w18=[0,-1]
w19=[1,-(1/4)]
w110=[1,(3/4)]
w111=[0,1]
w112=[-1,(3/4)]

#calculating the bias for the mouth
# a point on line 7 is [0.5,-(1/4)*0.5+1=0.875]
b17=-np.dot(w17,[0.5,0.875])

# a point on line 8 is [4,0.75]
b18=-np.dot(w18,[4,0.75])

# a point on line 9 is [7.5,(1/4)*7.5-1=0.875]
b19=-np.dot(w19,[7.5,0.875])

# a point on line 10 is [0.5,-(3/4)*0.5+1=0.625]
b110=-np.dot(w110,[0.5,0.625])

# a point on line 11 is [4,0.25]
b111=-np.dot(w111,[4,0.25])

# a point on line 12 is [7.5,(3/4)*7.5-5=0.625]
b112=-np.dot(w112,[7.5,0.625])



W1=[w17,w18,w19,w110,w111,w112]
B1=[[b17],[b18],[b19],[b110],[b111],[b112]]


#------------------------------------------------------------2nd layer AND------------------------------------------------------------

#for the mouth
#[1,1,1,1,1,1]*[[a6],[a7],[a8],[a19],[a10],[a11]]+b>=0 # to get '1'
#[1,1,1,1,1,1]*[[1],[1],[1],[1],[1],[1]]+b>=0
# 6+b>=0 -> b>=-6

#[1,1,1,1,1,1]*[[a6],[a7],[a8],[a9],[a10],[a11]]+b<0 # to get '0'
#[1,1,1,1,1,1]*[[1],[1],[1],[1],[1],[0]]+b<0
# 5+b<0 -> b<-5
# -6<=b<-5 -> b=-5.5

W2=[1,1,1,1,1,1]

B2=-5.5

#------------------------------------------------------------PLOTTING------------------------------------------------------------

xl7=np.arange(0,1.1,0.1)
yl7=-0.25*xl7+1

xl8=np.arange(1,7.1,0.1)
yl8=[]
for i in range(len(xl8)):
    yl8.append(0.75)

xl9=np.arange(7,8.1,0.1)
yl9=0.25*xl9-1

yl10=-0.75*xl7+1

yl11=[]
for i in range(len(xl8)):
    yl11.append(0.25)

yl12=0.75*xl9-5

plt.figure(1)

xp=np.linspace(0,8)
yp=np.linspace(0,1.5)

for x in xp:
    for y in yp:
        a1=np.dot(W1,[[x],[y]])+B1
        for i in range(len(a1)):
            a1[i]=hardlim(a1[i])
        a2=np.dot(W2,a1)+B2
        for i in range(len(a2)):
            a2[i]=hardlim(a2[i])
        
        if a2==1:
            plt.plot(x,y,'bo')
        else:
            plt.plot(x,y,'ro')


plt.plot(xl7,yl7,'k',xl8,yl8,'k',xl9,yl9,'k',xl7,yl10,'k',xl8,yl11,'k',xl9,yl12,'k')
plt.grid(True)

plt.show()