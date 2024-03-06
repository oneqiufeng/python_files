'''
求函数最小值点的 0.618 法
'''

import math as m

def p(a,b,eps):

    xmin = a;

    ya = f(a);

    yb = f(b);

    

    temp = ya-yb;

    if m.fabs(temp) >= eps:

        fr(a,b,temp);

    else:

        print(xmin);

def f(x):

    return x*x-m.sin(3*x)

def fr(a,b,temp):

    x1 = a+0.382*(a-b);

    x2 = a+0.618*(a-b);

    y1 = f(x1);

    y2 = f(x2);

    if y1 < y2:

        b = x2;

        yb = y2;

    else:

        a = x1;

        ya = y1;

    if m.fabs(temp) >= eps:

        p(a,b,eps);

    else:

        print(xmin);

a = float(input("请输入a:"));

b = float(input("请输入b:"));

eps = float(input("请输入eps:"));

p(a,b,eps);