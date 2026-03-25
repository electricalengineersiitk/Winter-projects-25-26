import math
import matplotlib.pyplot as plt
x1=[]
x2=[]
y=[]
with open('zombies_data.csv','r') as file:
    lines=file.read().strip().splitlines()
    for line in lines[1:]:
        pair=line.split(',')
        x1.append(float(pair[0]))
        x2.append(float(pair[1]))
        y.append(float(pair[2]))
    file.close()


n=len(x1)
mn=min(x1)
mx=max(x1)
for i in range(0,n,1):
    x1[i]=(x1[i]-mn)/(mx-mn)
mn=min(x2)
mx=max(x2)
for i in range(0,n,1):
    x2[i]=(x2[i]-mn)/(mx-mn)


alpha1=0.00000001
alpha2=0.0001
alpha3=0.00000001
m0=0
m1=0
m2=0
costplot=[]
for k in range(100000):
    cost=0
    for i in range(0,n,1):
        z=m0+m1*x1[i]+m2*x2[i]
        h=1/(1+math.exp(-z))
        m0=m0+alpha1*(y[i]-h)
        m1=m1+alpha2*(y[i]-h)*x1[i]
        m2=m2+alpha3*(y[i]-h)*x2[i]
        cost+=-y[i]*math.log(h)-(1-y[i])*math.log(1-h)
    costplot.append(cost/n)
x1.append(25)
mn=min(x1)
mx=max(x1)
x2.append(1)
tx1=(25-mn)/(mx-mn)
mn=min(x2)
mx=max(x2)
tx2=(1-mn)/(mx-mn)
q=m0+m1*tx1+m2*tx2
ans=1/(1+math.exp(-q))
print("The predicted probability of surviving a zombie attack with 25km/h speed and 1 ammo clip is:",round(ans,5))
print("The predicted outcome is:", "Survived" if ans>=0.5 else "Infected")

plt.plot([i for i in range(len(costplot))],costplot)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence using Gradient Descent')
plt.show()

temp=[i/10 for i in range(1,11,1)]
plt.plot(temp,[(m1*i-m0)/m2 for i in temp],color='red')
plt.xlabel('Speed (km/h)')
plt.ylabel('Ammo Clips')
plt.title('Decision Boundary for Zombie Survival')
plt.show()
