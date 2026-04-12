import math

x1=[0,0,1,1]
x2=[0,1,0,1]
t=[0,1,1,0]

w11=0.1
w12=0.1
w21=0.2
w22=0.2
w31=0.3
w32=0.3
b1=0.0
b2=0.0
b3=0.0
learnRate=0.1

for i in range(100000):
    for j in range(4):
        u=w11*x1[j]+w12*x2[j]+b1
        v=w21*x1[j]+w22*x2[j]+b2
        h1=1/(1+math.exp(-u))
        h2=1/(1+math.exp(-v))
        z=w31*h1+w32*h2+b3
        y=1/(1+math.exp(-z))

        g1=(y-t[j])*y*(1-y)
        w31=w31-learnRate*g1*h1
        w32=w32-learnRate*g1*h2
        b3=b3-learnRate*g1
        g21=g1*w31*h1*(1-h1)
        g22=g1*w32*h2*(1-h2)
        w11=w11-learnRate*g21*x1[j]
        w12=w12-learnRate*g21*x2[j]
        b1=b1-learnRate*g21
        w21=w21-learnRate*g22*x1[j]
        w22=w22-learnRate*g22*x2[j]
        b2=b2-learnRate*g22

for j in range(4):
    u=w11*x1[j]+w12*x2[j]+b1
    v=w21*x1[j]+w22*x2[j]+b2
    h1=1/(1+math.exp(-u))
    h2=1/(1+math.exp(-v))
    z=w31*h1+w32*h2+b3
    y=1/(1+math.exp(-z))
    print(x1[j],x2[j],1 if y>0.5 else 0)

