import  matplotlib.pyplot  as  plt 
import  numpy  as  np 
import  skfuzzy  as  fuzz

def triangle(ini,fin,a,b,c,step):  #--------------------------TRIANGLE-----------------------
  u=list(np.arange(ini,fin,step))
  
  #Fixes the universe list
  for i in range(len(u)):
    u[i]=round(u[i],1)

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
  u=list(np.arange(ini,fin,step))
  
  for i in range(len(u)):
    u[i]=round(u[i],1)

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


#Creating Universes and MF's
#FLAG MF
FLAG,F_NE=trapezoid(0,359,0,0,10,80,0.1)
FLAG,F_N=trapezoid(0,359,10,80,100,170,0.1)
FLAG,F_W=trapezoid(0,359,100,170,190,260,0.1)
FLAG,F_S=trapezoid(0,359,190,260,280,350,0.1)
FLAG,F_SE=trapezoid(0,359,280,350,359,359,0.1)

#GENERATOR MF
GENERATOR,G_NE=trapezoid(0,359,0,0,10,80,0.1)
GENERATOR,G_N=trapezoid(0,359,10,80,100,170,0.1)
GENERATOR,G_W=trapezoid(0,359,100,170,190,260,0.1)
GENERATOR,G_S=trapezoid(0,359,190,260,280,350,0.1)
GENERATOR,G_SE=trapezoid(0,359,280,350,359,359,0.1)

#SPEED MF
SPEED,S_LOW=trapezoid(0,100,0,0,10,35,0.1)
SPEED,S_MED=trapezoid(0,100,15,40,60,85,0.1)
SPEED,S_HIGH=trapezoid(0,100,65,90,100,100,0.1)

#POSITION MF
DIRECTION,D_CW=triangle(0,5,0,0,2.4,0.1)
DIRECTION,D_DM=triangle(0,5,2.3,2.5,2.7,0.1)
DIRECTION,D_ACW=triangle(0,5,2.6,5,5,0.1)

#Ask for the universe's value 
x=float(input("Introduce a Generator value: "))
y=float(input("Introduce a Flag value: "))

#finds the index of the universe where the input value is
G_i=GENERATOR.index(x)
F_i=FLAG.index(y)

#lists from the input value to the end of the universe
uxvals=GENERATOR[G_i:]
uyvals=FLAG[F_i:]

valGN=[]
valGNE=[]
valGW=[]
valGS=[]
valGSE=[]

valFN=[]
valFNE=[]
valFW=[]
valFS=[]
valFSE=[]

#create a list of the same value ONLY to graphic a line
for k in range(len(uxvals)):
  valGN.append(G_N[G_i])
  valGNE.append(G_NE[G_i])
  valGW.append(G_W[G_i])
  valGS.append(G_S[G_i])
  valGSE.append(G_SE[G_i])

for j in range(len(uyvals)):
  valFN.append(F_N[F_i])
  valFNE.append(F_NE[F_i])
  valFW.append(F_W[F_i])
  valFS.append(F_S[F_i])
  valFSE.append(F_SE[F_i])


# Fuzzy Inference System RULES
# First and Second Step
l1=min(G_N[G_i],F_N[F_i])
l2=min(G_N[G_i],F_NE[F_i])
l3=min(G_N[G_i],F_W[F_i])
l4=min(G_N[G_i],F_S[F_i]) 
l5=min(G_N[G_i],F_SE[F_i])

l6=min(G_NE[G_i],F_N[F_i]) 
l7=min(G_NE[G_i],F_NE[F_i])
l8=min(G_NE[G_i],F_W[F_i])
l9=min(G_NE[G_i],F_S[F_i])
l10=min(G_NE[G_i],F_SE[F_i])

l11=min(G_W[G_i],F_N[F_i])
l12=min(G_W[G_i],F_NE[F_i])
l13=min(G_W[G_i],F_W[F_i])
l14=min(G_W[G_i],F_S[F_i])
l15=min(G_W[G_i],F_SE[F_i])

l16=min(G_S[G_i],F_N[F_i])
l17=min(G_S[G_i],F_NE[F_i])
l18=min(G_S[G_i],F_W[F_i])
l19=min(G_S[G_i],F_S[F_i])
l20=min(G_S[G_i],F_SE[F_i])

l21=min(G_SE[G_i],F_N[F_i])
l22=min(G_SE[G_i],F_NE[F_i])
l23=min(G_SE[G_i],F_W[F_i])
l24=min(G_SE[G_i],F_S[F_i])
l25=min(G_SE[G_i],F_SE[F_i])



#Third Step
S_LOW_MAX=max(l1,l7,l10,l13,l19,l22,l25)
S_MED_MAX=max(l2,l3,l5,l6,l9,l11,l14,l17,l18,l20,l21,l24)
S_HIGH_MAX=max(l4,l8,l12,l15,l16,l23)

D_DM_MAX=max(l1,l7,l13,l19,l25)
D_CW_MAX=max(l2,l4,l5,l9,l10,l11,l12,l16,l18,l23,l24)
D_ACW_MAX=max(l3,l6,l8,l14,l15,l17,l20,l21,l22)





#Fourth and Fifth Step
C_S_LOW=[min(S_LOW_MAX,S_LOW[k]) for k in range(len(SPEED))]
C_S_MED=[min(S_MED_MAX,S_MED[k]) for k in range(len(SPEED))]
C_S_HIGH=[min(S_HIGH_MAX,S_HIGH[k]) for k in range(len(SPEED))]

C_D_DM=[min(D_DM_MAX,D_DM[k]) for k in range(len(DIRECTION))]
C_D_CW=[min(D_CW_MAX,D_CW[k]) for k in range(len(DIRECTION))]
C_D_ACW=[min(D_ACW_MAX,D_ACW[k]) for k in range(len(DIRECTION))]

#Sixth Step
finalCL_S=[max(C_S_LOW[k],C_S_MED[k],C_S_HIGH[k]) for k in range(len(SPEED))]
finalCL_D=[max(C_D_DM[k],C_D_CW[k],C_D_ACW[k]) for k in range(len(DIRECTION))]



fmFCL_S=np.asarray(finalCL_S)
fmFCL_D=np.asarray(finalCL_D)

FU_S=np.asarray(SPEED)
FU_D=np.asarray(DIRECTION)

cent_s=fuzz.defuzz(FU_S,fmFCL_S,'centroid')
cent_d=fuzz.defuzz(FU_D,fmFCL_D,'centroid')

"""

#FIGURE 1
fig=plt.figure(1)
fig.supylabel('Membership Degree')
fig.suptitle("Fuzzy Inference System")

plt.subplot(3,1,1)
plt.plot(X,A1,X,A2)
plt.title("A")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["A1=High Temperature","A2=Low Temperature"])

plt.subplot(3,1,2)
plt.plot(Y,B1,Y,B2)
plt.title("B")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["B1=Low Humidity","B2=High Humidity"])

plt.subplot(3,1,3)
plt.plot(Z,C1,Z,C2)
plt.title("C")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["C1=Low Speed","C2=High Speed"])
#plt.show()

#-------------------------------------------------------------------------------

#SECOND FIGURE
#FIGURE 2
fig2=plt.figure(2)
fig2.supylabel('Membership Degree')
fig2.suptitle("Fuzzy Inference System Rules")

plt.subplot(4,3,1)
plt.plot(X,A1,'#6b0e46',uxvals,valA1,'--k',x,A1[xi],"-s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["A1"])

plt.subplot(4,3,2)
plt.plot(Y,B1,"#104b51",uyvals,valB1,'--k',y,B1[yi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["B1"])

plt.subplot(4,3,3)
plt.plot(Z,C1,"#bc5090",Z,[la for i in range(len(Z))],"--k",Z,C1X,'x')
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["C1"])

plt.subplot(4,3,4)
plt.plot(X,A1,"#6b0e46",uxvals,valA1,'--k',x,A1[xi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["A1"])

plt.subplot(4,3,5)
plt.plot(Y,B2,"#ffa600",uyvals,valB2,'--k',y,B2[yi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["B2"])

plt.subplot(4,3,6)
plt.plot(Z,C1,"#bc5090",Z,[la for i in range(len(Z))],"--k",Z,C1X,'x')
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["C1"])

plt.subplot(4,3,7)
plt.plot(X,A2,"#aec200",uxvals,valA2,'--k',x,A2[xi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["A2"])

plt.subplot(4,3,8)
plt.plot(Y,B1,"#104b51",uyvals,valB1,'--k',y,B1[yi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["B1"])

plt.subplot(4,3,9)
plt.plot(Z,C2,"#50fdfd",Z,[lb for i in range(len(Z))],"--k",Z,C2X,'x')
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["C2"])

plt.subplot(4,3,10)
plt.plot(X,A2,"#aec200",uxvals,valA2,'--k',x,A2[xi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["A2"])

plt.subplot(4,3,11)
plt.plot(Y,B2,"#ffa600",uyvals,valB2,'--k',y,B2[yi],"s")
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["B2"])

plt.subplot(4,3,12)
plt.plot(Z,C2,"#50fdfd",Z,[lb for i in range(len(Z))],"--k",Z,C2X,'x')
plt.xlabel("Universe")
plt.grid(True)
plt.legend(["C2"])
#plt.show()

#-------------------------------------------------------------------------------
fig3=plt.figure(3)
plt.plot(Z,C1,"#bc5090",Z,C2,"#50fdfd",
         Z,C1X,'k',
         Z,C2X,'y',
         Z,finalCL,'x')
plt.axvline(cent, linewidth=0.5, color='m', label='Centroid')
plt.title("Fuzzy Inference Systems")
plt.ylabel("Membership Degree")
plt.xlabel("Universe ")
plt.grid(True)
plt.show()"""