import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

def hardlim(x):
    if (x<0).all():
        return 0
    else:
        return 1
    
def purelin(x):
    return x


# patterns
P=np.array([[1,1,2,2,3,3,4,4], [4,5,4,5,1,2,1,2]]) #matrix of patterns
Tp=np.array([0,0,0,0,1,1,1,1])
Ta=np.array([-1,-1,-1,-1,1,1,1,1])

[R,N]=P.shape 
S=1 # number of neurons
epoch=100 # number of epochs

#WEIGHTS AND BIAS FOR PERCEPTRON
Wp=np.random.rand(S,R) # matrix of weights
bp=np.random.rand(S,1) # matrix of bias

#WEIGHTS AND BIAS FOR ADALINE
Wa1=Wp
ba1=bp
Wa2=Wp
ba2=bp
Wa3=Wp
ba3=bp

R=(1/N)*np.dot(P,P.T)
eigen=np.linalg.eigvals(R)

#Alpha 
alpha1=1/(4*max(eigen))
alpha2=1/(8*max(eigen))
alpha3=1/(16*max(eigen))

#errors list
errorp=[]
errora1=[]
errora2=[]
errora3=[]

it=0
ue=[]
#Training 
for i in range(epoch):
    for j in range(N):
        ue.append(it)
        it=it+1

        x=P[:,j]
        
        #Percetron Training 
        ap=hardlim((np.dot(Wp,x)+bp))
        e_p=Tp[j]-ap
        errorp.append(e_p)
        Wp=Wp+e_p*x
        bp=bp+e_p

        #Adeline Training 1
        aa1=purelin((np.dot(Wa1,x)+ba1))
        e_a1=Ta[j]-aa1
        errora1.append(e_a1[0][0])
        Wa1=Wa1+alpha1*e_a1*x
        ba1=ba1+alpha1*e_a1

        #Adeline Training 2
        aa2=purelin((np.dot(Wa2,x)+ba2))
        e_a2=Ta[j]-aa2
        errora2.append(e_a2[0][0])
        Wa2=Wa2+alpha2*e_a2*x
        ba2=ba2+alpha2*e_a2

        #Adeline Training 3
        aa3=purelin((np.dot(Wa3,x)+ba3))
        e_a3=Ta[j]-aa3
        errora3.append(e_a3[0][0])
        Wa3=Wa3+alpha3*e_a3*x
        ba3=ba3+alpha3*e_a3

X = np.arange(0,5,0.2)
Y = np.arange(0,5,0.2)

print(bp)
print(ba1)
print(ba2)
#--------------------------------------------------------------PLOT PATTERNS
a=P.tolist()

fig=plt.figure(figsize=(10,10),tight_layout=True)
fig.suptitle("PRACTICE 1 PERCEPTRON VS ADALINE")
plt.subplot(2,2,1)
for i in range(len(a[0])):
    if i <=3:
        plt.plot(a[0][i],a[1][i],'go',markersize=5)
    else:
        plt.plot(a[0][i],a[1][i],'ro',markersize=5)

#--------------------------------------------------------------LIMIT LINE ALPHA1
Wpp=Wp[0].tolist()
bpp=bp[0].tolist()

limlineP=[]
for x in X:
    limlineP.append(-bpp[0]/Wpp[1]-Wpp[0]/Wpp[1]*x)
plt.plot(X,limlineP,'cv-', markersize=3,label='Perceptron')

#--------------------------------------------------------------LIMIT LINE ALPHA1
Wa1p=Wa1[0].tolist()
ba1p=ba1[0].tolist()

limlineA1=[]
for x in X:
    limlineA1.append(-ba1p[0]/Wa1p[1]-Wa1p[0]/Wa1p[1]*x)
plt.plot(X,limlineA1,'k--', markersize=3,label='Alpha 1')

#--------------------------------------------------------------LIMIT LINE ALPHA2
Wa2p=Wa2[0].tolist()
ba2p=ba2[0].tolist()

limlineA2=[]
for x in X:
    limlineA2.append(-ba2p[0]/Wa2p[1]-Wa2p[0]/Wa2p[1]*x)
plt.plot(X,limlineA2,'b-o', markersize=3,label='Alpha 2')

#--------------------------------------------------------------LIMIT LINE ALPHA3
Wa3p=Wa3[0].tolist()
ba3p=ba3[0].tolist()

limlineA3=[]
for x in X:
    limlineA3.append(-ba3p[0]/Wa3p[1]-Wa3p[0]/Wa3p[1]*x)
plt.plot(X,limlineA3,'g-s', markersize=3,label='Alpha 3')
plt.xlabel('P1')
plt.ylabel('P2')
plt.title('P1 vs P2')
plt.legend(loc="lower left")
plt.grid(True)
 
#--------------------------------------------------------------PLOT PERCETRON ERROR

plt.subplot(2,2,3)
plt.plot(ue,errorp,'r-o',markersize=1,label='Perceptron')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('PERCEPTRON Error')
plt.legend()
plt.grid(True)
#--------------------------------------------------------------PLOT ADALINE ERROR

plt.subplot(2,2,4)
plt.plot(ue,errora1,'b-o',markersize=1,label='Alpha 1')
plt.plot(ue,errora2,'g-o',markersize=1,label='Alpha 2')
plt.plot(ue,errora3,'k-o',markersize=1,label='Alpha 3')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.title('PERCEPTRON Error')
plt.grid(True)

plt.show()


#--------------------------------------------------------------TESTING
bear = mpimg.imread('bear.jpg')
rabbit = mpimg.imread('rabbit.jpg')


condition="y"

while condition=="y":
        
    weight = int(input("Introduce the weight: "))
    ears = int(input("Introduce the ears: "))

    pattern=np.array([[weight],[ears]])
    ap=hardlim((np.dot(Wp,pattern)+bp))
    plt.figure()
    if ap==1:
        print("The perceptron says that it is a bear")
        plt.subplot(1,2,1)
        plt.imshow(bear)
        plt.title("Perceptron")
    else:
        print("The perceptron says that it is a rabbit")
        plt.subplot(1,2,1)
        plt.imshow(rabbit)
        plt.title("Perceptron")

    aa=purelin((np.dot(Wa2,pattern)+ba2))
    if aa>0:
        print("The Adaline says that it is a bear")
        plt.subplot(1,2,2)
        plt.imshow(bear)
        plt.title("Adaline")
    else:
        print("The Adaline says that it is a rabbit")
        plt.subplot(1,2,2)
        plt.imshow(rabbit)
        plt.title("Adaline")

    plt.show()
    condition=input("Do you want to continue? (y/n): ") 

print("BYE!")
