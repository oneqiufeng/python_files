# 计算和式 1！+2！+3！+n！大于 1010 时，n 最小是多少？

def func(n):

    if n==0 or n==1:

        return 1

    else:

        return(n*func(n-1))

sum=0

for i in range(1,1000):

    sum=sum+func(i)

    if sum>10**10:

        print(i)

        break