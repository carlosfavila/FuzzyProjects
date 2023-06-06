import matplotlib.pyplot as plt
import numpy as np

def hardlim(x):
    if (x<0).all():
        return 0
    else:
        return 1

#------------------------------------------------------------LEFT EYE------------------------------------------------------------
#proposing 3 points as the triangle vertices
#P1=[2,2]
#P2=[1,1]
#P3=[3,1]

#so, for this we have 3 lines (neurons):
# Yl1 - 2= (2-1/2-1)*(X-2) -> Yl1 = X
# Yl2 - 2= (2-1/2-3)*(x-2) -> Yl2 = -X+4
# YL3 = 1

#So, the weight line equations are:
# Yw11 = -X
# Yw12 = X
# Yw13 = inf

# so, we want the line of the weight 1 points at the middle of the three lines
# proposing x = 1, with Yw11=-x we get Yw11 = -1

# and we also want the line of the weight 2 points at the middle of the thre lines
# proposing x = -1, with Yw12=x we get Yw12 = -1

# and we also want the line of the weight 3 points at the middle of the thre lines
# in this case, the weight 3 is a vertical line, so we can propose x=0 and in Yw13=1

w11=[1,-1]
w12=[-1,-1]
w13=[0,1]

# calculating bias for lef eye
# a point on line 1 is [0,0]
b11=-np.dot(w11,[0,0])

# a point on line 2 is [0,4]
b12=-np.dot(w12,[0,4])

# a point on line 3 is [0,1]
b13=-np.dot(w13,[0,1])

#------------------------------------------------------------RIGHT EYE------------------------------------------------------------
#proposing 3 points as the triangle vertices
#P4=[6,2]
#P5=[5,1]
#P6=[7,1]

#so, for this we have 3 lines (neurons):
# Yl4 - 2= (2-1/6-5)*(X-6) -> Yl4 = X-4
# Yl5 - 2= (2-1/6-7)*(x-6) -> Yl5 = -X+8
# Yl6 = 1 = Yl3

#So, the weight line equations are:
# Yw14 = -X
# Yw15 = X
# Yw16 = inf

# so, we want the line of the weight 4 points at the middle of the three lines
# proposing x = 1, with Yw14=-x we get Yw14 = -1

# and we also want the line of the weight 5 points at the middle of the thre lines
# proposing x = -1, with Yw15=x we get Yw15 = -1

# and we also want the line of the weight 6 points at the middle of the thre lines
# in this case, the weight 3 is a vertical line, so we can propose x=0 and in Yw16=1

# we can see that we have the same weights as the left eye, so we can use the same weights

w14,w15,w16=w11,w12,w13
# calculating bias for right eye

# a point on line 1 is [0,-4]
b14=-np.dot(w14,[0,-4])

# a point on line 2 is [0,8]
b15=-np.dot(w15,[0,8])

# a point on line 3 is [0,1]
b16=-np.dot(w16,[0,1])

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

# a point on line 8 is [1,0.75]
b18=-np.dot(w18,[4,0.75])

# a point on line 9 is [7.5,(1/4)*7.5-1=0.875]
b19=-np.dot(w19,[7.5,0.875])

# a point on line 10 is [0.5,-(3/4)*0.5+1=0.625]
b110=-np.dot(w110,[0.5,0.625])

# a point on line 11 is [4,0.25]
b111=-np.dot(w111,[4,0.25])

# a point on line 12 is [7.5,(3/4)*7.5-5=0.625]
b112=-np.dot(w112,[7.5,0.625])



W1=[w11,w12,w13,w14,w15]
B1=[[b11],[b12],[b13],[b14],[b15]]


#------------------------------------------------------------2nd layer AND------------------------------------------------------------
#W2=[[1,1,1,0,0,0,0,0,0,0,0], # this line is the sum of the weights of the left eye
#    [0,0,1,1,1,0,0,0,0,0,0],  # this line is the sum of the weights of the right eye
#    [0,0,0,0,0,1,1,1,1,1,1]] # this line is the sum of the weights of the mouth

#W*P+b>=0 MINIMUM CONDITION TO GET '1'
#[1,1,1,0,0,0,0,0,0,0,0]*[[a1],[a2],[a3],[a4],[a5],[a6],[a7],[a8],[a9],[a10],[a11]]+b>=0
#[0,0,1,1,1,0,0,0,0,0,0]
#[0,0,0,0,0,1,1,1,1,1,1]

#W*P+b<0 MAXIMUM CONDITION TO GET '0'
#[1,1,1,0,0,0,0,0,0,0,0]*[[a1],[a2],[a3],[a4],[a5],[a6],[a7],[a8],[a9],[a10],[a11]]+b<0
#[0,0,1,1,1,0,0,0,0,0,0]
#[0,0,0,0,0,1,1,1,1,1,1]

# for the right eye
#[1,1,1]*[[a1],[a2],[a3]]+b>=0 # to get '1'
#[1,1,1]*[[1],[1],[1]]+b>=0
# 3+b>=0 -> b>=-3
#[1,1,1]*[[a1],[a2],[a3]]+b<0 # to get '0'
#[1,1,1]*[[1],[1],[0]]+b<0 
# 2+b<0 -> b<-2
# -3<=b<-2 -> b=-2.5

# for the left eye
#[1,1,1]*[[a3],[a4],[a5]]+b>=0 # to get '1'
#[1,1,1]*[[1],[1],[1]]+b>=0
# 3+b>=0 -> b>=-3
#[1,1,1]*[[a3],[a4],[a5]]+b<0 # to get '0'
#[1,1,1]*[[1],[1],[0]]+b<0
# 2+b<0 -> b<-2
# -3<=b<-2 -> b=-2.5

#for the mouth
#[1,1,1,1,1,1]*[[a6],[a7],[a8],[a19],[a10],[a11]]+b>=0 # to get '1'
#[1,1,1,1,1,1]*[[1],[1],[1],[1],[1],[1]]+b>=0
# 6+b>=0 -> b>=-6

#[1,1,1,1,1,1]*[[a6],[a7],[a8],[a9],[a10],[a11]]+b<0 # to get '0'
#[1,1,1,1,1,1]*[[1],[1],[1],[1],[1],[0]]+b<0
# 5+b<0 -> b<-5
# -6<=b<-5 -> b=-5.5

W2=[[1,1,1,0,0],
    [0,0,1,1,1]]

B2=[[-2.5],[-2.5]]

#------------------------------------------------------------3rd layer OR------------------------------------------------------------

W3=[1,1] # all inputs have the same weight (same importance)

#W*P+b>=0 MINIMUM CONDITION TO GET '1'
#[1,1,1]*[[1],[0],[0]]+b>=0 or [1,1,1]*[[0],[1],[0]]+b>=0 or [1,1,1]*[[0],[0],[1]]+b>=0
# 1+b>=0 -> b>=-1

#W*P+b<0 MAXIMUM CONDITION TO GET '0'
#[1,1,1]*[[0],[0],[0]]+b<0 
# 0+b<0 -> b<0

#-1<=b<0 -> b=-0.5

B3=[-0.5]

#------------------------------------------------------------PLOTTING------------------------------------------------------------
xl1=np.arange(1,2.1,0.1)
yl1=xl1

xl2=np.arange(2,3.1,0.1)
yl2=-xl2+4

xl3=np.arange(1,3.1,0.1)
yl3=[]
for i in range(len(xl3)):
    yl3.append(1)

xl4=np.arange(5,6.1,0.1)
yl4=xl4-4

xl5=np.arange(6,7.1,0.1)
yl5=-xl5+8

xl6=np.arange(5,7.1,0.1)
yl6=[]

for i in range(len(xl6)):
    yl6.append(1)


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
yp=np.linspace(0,3)

for x in xp:
    for y in yp:
        a1=np.dot(W1,[[x],[y]])+B1
        for i in range(len(a1)):
            a1[i]=hardlim(a1[i])
        a2=np.dot(W2,a1)+B2
        for i in range(len(a2)):
            a2[i]=hardlim(a2[i])
        a3=hardlim(np.dot(W3,a2)+B3)
        
        if a3==1:
            plt.plot(x,y,'bo')
        else:
            plt.plot(x,y,'ro')


plt.plot(xl1,yl1,'k',xl2,yl2,'k',xl3,yl3,'k',xl4,yl4,'k',xl5,yl5,'k',xl6,yl6,'k',xl7,yl7,'k',xl8,yl8,'k',xl9,yl9,'k',xl7,yl10,'k',xl8,yl11,'k',xl9,yl12,'k')
plt.grid(True)

plt.show()