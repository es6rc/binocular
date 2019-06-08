#!/usr/bin/env python

import math
from sys import exit

# The ik fn. is used for specific inverse kinematic for the head-neck sys.
def ik(Xw, Yw, Zw):
    # if type(Xw)== type(Yw) == type(Zw) == Symbol:
    if Yw == 0:
        if Xw>0:
            q1 = -math.pi/2
        elif Xw<0:
            q1 = math.pi/2
        else:
            q1 = 0

    else:
        q1 = -math.atan(Xw/Yw)
    theta1 = -math.pi/2+q1
    s1 = math.sin(theta1)
    c1 = math.cos(theta1)
    
    a = -s1*Xw+c1*Yw+0.7
    #print(a)
    b = -c1*Xw-s1*Yw
    #print(b)
    if (Xw == 0 and Yw == 0) or b == 0:
        theta3 = 3*math.pi/2
    else:
        alpha3 = math.atan(a/b)
        if alpha3 < 0:
            exit('motor2 execeeds limit')
        theta3 = math.pi - alpha3
    #print(theta3)
    q3 = theta3 - math.pi
    s3 = math.sin(theta3)
    c3 = math.cos(theta3)
    c = (s1*s3-c1*c3)*Xw-(s1*c3+s3*c1)*Yw-0.7*s3
    #print(temp0)
    d = Zw - 1.45
    #print(c,d)
    if c == 0:
        if d < 0:
            beta = -math.pi/2
        elif d > 0:
            beta = math.pi/2
        else:
            exit('Computational Error!! Examine your equations!!')
    elif c < 0:
        if d != 0:
            beta = math.pi + math.atan(d/c)
        else:
            beta = math.pi
    else:
        beta = math.atan(d/c)
    
    den = (c**2+d**2)**0.5

    temp1 = math.asin(0.2/den)
    #temp1 = round(temp1, 8)
    #temp2 = round(e, 8)
    theta4 = math.pi - temp1 - beta
    q4 = theta4
    #q_3 = -math.atan(0.7/(Yw**2+Xw**2)**0.5)
    #print(math.degree(q_3))
    return q1, q3, q4 
    #math.degrees(q1), math.degrees(q3), math.degrees(q4)




# print(solvethetas(-1, 7, 1.65))

