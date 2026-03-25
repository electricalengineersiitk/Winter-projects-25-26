import matplotlib.pyplot as plt
x=[]
y=[]
with open('housing_prices.csv','r') as file:
    lines=file.read().strip().splitlines()
    for line in lines[1:]:
        pair=line.split(',')
        x.append(float(pair[0]))
        y.append(float(pair[1]))
    file.close()
alpha2=0.00000000001
alpha1=0.00001
m=0
b=0
n=len(x)
for k in range(100000):
    for i in range(0,n,1):
        h=m*x[i]+b
        m=m+alpha2*(y[i]-h)*x[i]
        b=b+alpha1*(y[i]-h)
ans=m*2500+b
print("The predicted price of a house with 2500 sq.ft area using GDA is:",round(ans,2))
xplot=[i for i in range(1000,3500,100)]
yplot=[m*i+b for i in xplot]
plt.plot(xplot,yplot,color='red')
plt.xlabel('Area (sq.ft)')
plt.ylabel('Price')
plt.title('Housing Prices Prediction using GDA')
plt.show()
