# 请编程找出从 1 到 1000 中能被 41 整除但不能被 7 整除的所有整数， 并计算这些整数之和。

sum=0

for i in range (1,1001):

    if i%41==0 and i%7!=0:

        print(i)

        j=i

    else:

        j=0

    sum=sum+j

print(sum)