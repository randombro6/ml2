#mean
def mean(list):
    
    sum=0
    for i in list:
        sum=sum+i
        
    mean=sum/len(list)
    return mean


list=(10,20,30,40,50,60,70)
print("mean",mean(list))
mean 40.0

#median
l1=(10,20,30,40,50,60,70)
i=int(len(l1)/2)
if(i%2==0):
    res=(l1[i]+l1[i+1])/2
else:
    res=l1[i]
print("median",res)    
median 40

#mode
l1=(10,20,30,30,40,50,60,70)
d={}
for i in l1:
    if i in d:
        d[i]=d[i]+1
    else:
        d[i]=1
max=0
for i in d:
    if(d[i]>max):
        max=d[i]
        ans=i
print("MODE",ans)        
        
        
MODE 30

#variance
def var(l):
    ans=mean(l)
    
    sum=0
    for i in l:
        sum+=(ans-i)**2
        
    return sum/len(l)
l=(10,20,30,40,50,60,70)
print("variance",var(l))
variance 400.0

#standard deviation
def sd(l):
    temp=var(l)
    return temp**0.5

l=(10,20,30,40,50,60,70)
print("standard deviation",sd(l))
standard deviation 20.0


#normalization
def nor(l):
    min=l[0]
    max=l[0]
    for i in l:
        if(i>max):
            max=i
        elif(i<min):
            min=i
            
    for i in l:
        print((i-min)/(max-min))
        
l=(10,20,30,40,50,60,70)
nor(l)
0.0
0.16666666666666666
0.3333333333333333
0.5
0.6666666666666666
0.8333333333333334
1.0

#standardization
def std(l):
    for i in l:
        print((i-mean(l))/sd(l))
l=(10,20,30,40,50,60,70)
std(l)              
            
-1.5
-1.0
-0.5
0.0
0.5
1.0
1.5