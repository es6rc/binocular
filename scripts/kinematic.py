
# coding: utf-8

# In[1]:


from sympy import *
import math
from IPython.display import Latex


# In[2]:



#					##########TFs&PARAMs##########
#
#	| TRANSFORMATION |	 THETA	 |	 ALPHA	 |	 D	 |	 R	 |
#	| -------------  | --------- | --------- | ----- | ----- |
#	|		Tw0		 |	  N/A	 |	  N/A	 |	N/A	 |	N/A	 |
#	|		T01		 |	 theta1	 |	 alpha1	 |  d1   |  r1   |
#	|		T12		 |	 theta2	 |	 alpha2	 |  d2   |  r2   |
#	|		T23		 |	 theta3	 |	 alpha3	 |  d3   |  r3   |
#	|		T34		 |	 theta4	 |	 alpha4	 |  d4   |  r4   |
#	|		T4L		 |	  N/A	 |	  N/A	 |	N/A	 |	N/A	 |
#	|		T15		 |	 theta2	 |	 alpha5	 |  d5   |  r5   |
#	|		T56		 |	 theta6	 |	 alpha6	 |  d6   |  r6   |
#	|		T67		 |	 theta7	 |	 alpha7	 |  d7   |  r7   |
#	|		T7R		 |	  N/A	 |	  N/A	 |	N/A	 |	N/A	 |
#
#			T3L = T34 * T4L & T6R = T67 * T7R
#			can be seen as single matrix
#
#
#			##########PARAMETERS VALUEs##########
#	### gazebo simulation model kinematic
#	|-pi/2-theta1|	  var	 | |	alpha1	 |	  -90	 |
#	|     d1     |	  0.1	 | |      r1     |     0	 |
#
#	| pi-theta2  |	  var	 | |	alpha2	 |	  -90	 |
#	|     d2     |	 -0.7	 | |      r2     |     0	 |
#
#	| pi-theta3  |	  var	 | |	alpha3	 |	  -90	 |
#	|     d3     |	  0.45	 | |      r3     |     0	 |
#
#	|	theta4	 |	  var	 | |	alpha4	 |	   0	 |
#	|     d4     |	   0	 | |      r4     |     0	 |
#
#	|	theta5=2 |	  var	 | |	alpha5	 |	  -90	 |
#	|     d5     |	  0.7	 | |      r5     |     0	 |
#
#	|	theta6=3 |	  var	 | |	alpha6	 |	  -90	 |
#	|     d6     |	  0.45	 | |      r6     |     0	 |
#
#	|	theta7=4 |	  var	 | |	alpha7	 |	   0	 |
#	|     d7     |	   0	 | |      r7     |     0	 |

#	!! Denavit-Hartenberg (DH) Transformations !!
#
#	T(n-1)<-n = 
<<<<<<< HEAD
#  +-                                  -+     +-       -+
#  |   ctn       stn       0  |   -rn   |     |     |   |
#  | -stn*can  ctn*can    san | -dn*san |     |  R  | t |
#  |    0     -ctn*san    can | -dn*can |  =  |     |   |
#  | -----------------------+-----------|     |-----+---|
#  |    0         0        0  |    1    |     |0 0 0| 1 |
#  +-                                  -+     +-       -+
=======
#	+-								 -+	   +-	    -+
#	| ctn  -stn*can  stn*san | rn*ctn |	   |	 |   |
#	| stn   ctn*can -ctn*san | rn*stn |	   |  R  | t |
#	|  0	  san  	   can   |   dn   |  = |     |   |
#	| -----------------------+--------|    |-----+---|
#	|  0	   0		0	 |   1	  |    |0 0 0| 1 |
#	+-								 -+    +-		-+
>>>>>>> 4dbaf65253bb3e71c5b366b09bb3f0ed27a8af79
# 	ctn = cos(THETAn) can =cos(ALPHAn) stn = sin(THETAn) san = sin(ALPHAn)
#
#	T(n-1)->n = 
#	+-								 -+	   +-	    -+
#	| ctn  -stn*can  stn*san | rn*ctn |	   |	 |   |
#	| stn   ctn*can -ctn*san | rn*stn |	   |  R  | t |
#	|  0	  san  	   can   |   dn   |  = |     |   |
#	| -----------------------+--------|    |-----+---|
#	|  0	   0		0	 |   1	  |    |0 0 0| 1 |
#	+-								 -+    +-		-+
# 	ctn = cos(THETAn) can =cos(ALPHAn) stn = sin(THETAn) san = sin(ALPHAn)
#
#
#	!!Translation&Yaw->Pitch->Roll Transformations !!
#				  phi->theta->psi
#
#	T(n-1)<-n = 
#	+-															-+
#	|ct*cpsi  sphi*st*cpsi-cphi*spsi  cphi*st*cpsi+sphi*spsi | x |
#	|ct*spsi  sphi*st*spsi+cphi*cpsi  cphi*st*spsi-sphi*cpsi | y |
#	|  -st            sphi*ct                cphi*ct         | z |
#	| -------------------------------------------------------+---|
#	|    0                0                      0           | 1 |
#	+-															-+
#	c -> cos() s -> sin()



# In[3]:


def dhtf(tf_type, n, alphan, dn, rn,thetan):

	alpha = alphan
	d = dn
	r = rn
	ca = round(math.cos(alpha), 2)
	sa = round(math.sin(alpha), 2)
	if tf_type == 'symbolic':
		theta = symbols('t%d'%n)
		T = Matrix([
			[cos(theta), -sin(theta)*ca, sin(theta)*sa, r*cos(theta)],
			[sin(theta), cos(theta)*ca, -cos(theta)*sa, r*sin(theta)],
			[0, sa, ca, d],
			[0, 0, 0, 1]])
	if tf_type == 'constant':
		#theta = 0
		ct = round(math.cos(thetan),2)
		st = round(math.sin(thetan),2)
		T = Matrix([
			[ct, -st*ca, st*sa, r*ct],
			[st, ct*ca, -ct*sa, r*st],
			[0, sa, ca, d],
			[0, 0, 0, 1]])
	return T


# In[4]:


def invdhtf(tf_type, n ,alphan, dn, rn, thetan):
    alpha = alphan
    d = dn
    r = rn
    ca = round(math.cos(alpha), 2)
    sa = round(math.sin(alpha), 2)
    if tf_type == 'symbolic':
        theta = symbols('t%d'%n)
        T = Matrix([
            [cos(theta), sin(theta), 0, -r],
            [-ca*sin(theta), cos(theta)*ca, sa, -d*sa],
            [sa*sin(theta), -sa*cos(theta), ca, -d*ca],
            [0, 0, 0, 1]])
    if tf_type == 'constant':
        #theta = 0
        ct = round(math.cos(thetan),2)
        st = round(math.sin(thetan),2)
        T = Matrix([
            [ct, st, 0, -r],
            [-ca*st, ct*ca, sa, -d*sa],
            [sa*st, -sa*ct, ca, -d*ca],
            [0, 0, 0, 1]])
    return T


# In[5]:


def rttf(x,y,z,phi,theta,psi):
    ct = round(math.cos(theta), 2)
    cphi = round(math.cos(phi), 2)
    cpsi = round(math.cos(psi), 2)
    st = round(math.sin(theta), 2)
    sphi = round(math.sin(phi), 2)
    spsi = round(math.sin(psi), 2)
    T = Matrix([
        [ct*cpsi, sphi*st*cpsi-cphi*spsi, cphi*st*cpsi+sphi*spsi, x],
        [ct*spsi, sphi*st*spsi+cphi*cpsi, cphi*st*spsi-sphi*cpsi, y],
        [-st, sphi*ct, cphi*ct,z],
        [0, 0, 0, 1]
    ])
    return T


# In[6]:


def pw(p_type, x, y, z):
    if p_type == 'symbolic':
        Xw = symbols('XL')
        Yw = symbols('YL')
        Zw = symbols('ZL')
        p = Matrix([Xw, Yw, Zw, 1])
    if p_type == 'constant':
        p = Matrix([x, y, z, 1])
    return p


# In[7]:


def latexmtx(tf):
    startl = '\\begin{bmatrix}'
    endl = '\\end{bmatrix}'
    temp1 = str(latex(tf))
    temp2 = temp1.replace('\\left[\\begin{matrix}',startl)
    line = temp2.replace('\\end{matrix}\\right]',endl)
    return line


# In[8]:


PI_2 = math.pi/2
r = 0 # All r here are "0"
alpha1 = -PI_2
d1 = 0.1
alpha2 = -PI_2
d2 = 0.7
alpha3 = -PI_2
d3 = 0.45
alpha4 = 0
d4 = 0

# n->0 Transformation
Tw0 = rttf(0, 0, 0.9, 0, 0, 0)
T01 = dhtf('symbolic', 1, alpha1, d1, r, -PI_2)
T12 = dhtf('constant', 2, alpha2, -d2, r, math.pi)
T23 = dhtf('symbolic', 3, alpha3, d3, r, math.pi)
T34 = dhtf('symbolic', 4, alpha4, d4, r, 0)
T4L = rttf(0, -0.2, 0, 0, PI_2, math.pi)
TwL = Tw0 * T01 * T12 * T23 * T34 * T4L

# Inverse(0->n) Transformation

T0w = Tw0.inv()
T10 = invdhtf('symbolic', 1, alpha1, d1, r,-PI_2)
T21 = invdhtf('constant', 2, alpha2, -d2, r, math.pi)
T32 = invdhtf('symbolic', 3, alpha3, d3, r, math.pi)
T43 = invdhtf('symbolic', 4, alpha4, d4, r, 0)
TL4 = T4L.inv()
TLw = TL4 * T43 * T32 * T21 * T10 * T0w

# Point in the World Frame
Pw = pw('constant',0,0,symbols('Zw'))
PL = TLw * Pw
# print(PL)
Latex(latexmtx(Pw))
# print(Pw)


# In[10]:


print('Tw0')
Latex(latexmtx(Tw0))


# In[11]:


print('T0w')
Latex(latexmtx(T0w))


# In[12]:


print('T01')
Latex(latexmtx(T01))
#Latex(latexmtx(T01.inv()))


# In[13]:


print('T10')
Latex(latexmtx(T10))


# In[14]:


print('T12')
Latex(latexmtx(T12))
#Latex(latexmtx(T12.inv()))


# In[15]:


print('T21')
Latex(latexmtx(T21))


# In[16]:


print('T23')
Latex(latexmtx(T23))
#Latex(latexmtx(T23.inv()))


# In[17]:


print('T32')
Latex(latexmtx(T32))


# In[18]:


print('T34')
Latex(latexmtx(T34))
Latex(latexmtx(T34.inv()))


# In[19]:


print('T43')
Latex(latexmtx(T43))


# In[20]:


print('T4L')
Latex(latexmtx(T4L))
#Latex(latexmtx(T4L.inv()))


# In[21]:


print('TL4')
Latex(latexmtx(TL4))


# In[9]:


print('TwL')
Latex(latexmtx(TwL))
#Latex(latexmtx(TwL.inv()))


# In[10]:


print('TLw')
Latex(latexmtx(TLw))


# In[122]:


PL = TLw * Pw

