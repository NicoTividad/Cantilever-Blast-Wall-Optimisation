import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

pi = np.pi
k = 3750000       #min 1e6, max 7e6                             #best = 3750000
m = 650       #min 200, max 1200                              #best = 650
x = 4.770801015182476       #min 0, max 10                      #best = 4.770801015182476
t = 0.5                                                    #best cost = 24278.69

tstep = 0.001
lx = 0
ux = 10

#functions for finding Fpeak and td ------------function 2
def get_Fpeak(x):
    Fpeak = 1000+9*(x**2)-183*x
    return Fpeak*1000

def get_td(x):
    td = 20-0.12*(x**2)+4.2*x
    return td/1000

#functions for associated costs --------------functions 3,4,5
def costsstiffness(k):
    ck = 900+825*((k/1e6)**2)-1725*(k/1e6)
    return(ck)
def costmass(m):
    cm = 10*m-2000
    return(cm)
def purchaseland(x):
    cx = 2400*((x**2)/4)
    return(cx)

def total_cost(x, k, m):
    ck = costsstiffness(k)
    cm = costmass(m)
    cx = purchaseland(x)
    ctotal = ck + cm + cx
    return ctotal
    

#functions for calculating z ----------------function 1

def get_z(t, x, k, m):
    omega = np.sqrt( k / m)
    Fpeak = get_Fpeak(x)
    td = get_td(x)
    if t <= td:
        z = Fpeak / k * (1 - np.cos(omega * t)) + Fpeak / (k * td) * (np.sin(omega * t) / omega - t)
    else:
        z = Fpeak / (k * omega * td) * (np.sin(omega * t) - np.sin(omega * (t -td))) - Fpeak / k * np.cos(omega * t)
    return z

def get_z_overtime(x, k, m, tstep):
    omega = np.sqrt(k/m)
    T = 2 * np.pi / omega
    t = np.arange(0, 2 * T, tstep)
    z = []
    for ti in t:
        zi = get_z(ti, x, k, m)
        z.append(zi)
    return z, t

def get_max_z(x, k, m, tstep):
  z, t = get_z_overtime(x, k, m, tstep)
  return np.max(np.abs(z))

#--------------------------------------------bisector method
def get_opt_z(lx,ux,k,m,tstep):
    lower_x = lx
    upper_x = ux
    middle_x = (lower_x+upper_x)/2
    q = get_max_z(middle_x,k,m,tstep)
    return q, middle_x

#---------------------------------------------coding section-----------------------------------------


x = 0
k = 1e6
m = 200
tstep = 0.001
print(get_max_z(x, k, m, tstep))



k = 4e6
m = 400
tstep = 0.001
lx = 0
ux = 10

for i in range(1,100):
    zmax, x = get_opt_z(lx, ux, k, m, tstep)
    print("\nz value: " + str(zmax))
    print("middle x value: " + str(x))
    if zmax < 0.1:
        ux = x
        print("lower: " + str(lx), "\nupper: " + str(ux))
    else:
        lx = x
        
x = np.around(x,3)
print("x value: " + str(x))

xlist = []
klist = []
mlist = []
costlist = []
for k in range(1000000, 7000001, 125000):
    for m in range(200, 1201, 50):
        lx = 0
        ux = 10
        for i in range(1,100):
            cost = total_cost(x, k, m)
            cost = np.round(cost,2)
            xlist.append(x)
            klist.append(k)
            mlist.append(m)
            costlist.append(cost)


# xlist = []
# klist = []
# mlist = []
# costlist = []
# for k in range(1000000, 7000001, 125000):
#     for m in range(200, 1201, 50):
#         print("\nk value: " + str(k))
#         print("m value: " +str(m))
#         lx = 0
#         ux = 10
#         for i in range(1,100):
#             zmax, x = get_opt_z(lx, ux, k, m, tstep)
#             #print("\nz value: " + str(zmax))
#             #print("middle x value: " + str(x))
#             if zmax < 0.1:
#                 ux = x
#                 #print("lower: " + str(lx), "\nupper: " + str(ux))
#             else:
#                 lx = x
#                 #print("lower: " + str(lx), "\nupper: " + str(ux))
#         #x = np.around(x,3)
#         print("x value: " + str(x))
#         print("z value: " + str(zmax))
#         cost = total_cost(x, k, m)
#         cost = np.round(cost,2)
#         print("total costs are " + str(cost))
#         xlist.append(x)
#         klist.append(k)
#         mlist.append(m)
#         costlist.append(cost)

# print(xlist)
# print(klist)
# print(mlist)
# print(costlist)
# cheapcost = min(costlist)
# print("cheapest total: " + str(cheapcost))
# cheapcostindex = costlist.index(cheapcost)
# print("at index " + str(cheapcostindex))

# xvalue = xlist[cheapcostindex]
# kvalue = klist[cheapcostindex]
# mvalue = mlist[cheapcostindex]
# costvalue = costlist[cheapcostindex]

# print("x value: " + str(xvalue))
# print("k value: " + str(kvalue))
# print("m value: " + str(mvalue))
# print("costing total: " + str(costvalue))

# zvalue = get_max_z(xvalue, kvalue, mvalue, tstep)
# print("z gives: "+ str(zvalue))




# #---------------------------------------------------graph drawing--------------------------------------
# fig = plt.figure(figsize=(9,7))
# axis = fig.add_subplot(111, projection = '3d')
# zdata = xlist
# xdata = klist
# ydata = mlist
# graph = axis.scatter3D(xdata,ydata,zdata, c=zdata, cmap = "plasma")
# axis.set_title( "a graph that varies k against m against x")
# axis.set_xlabel(" k (MN/m)")
# axis.set_ylabel(" m (kg)")
# axis.set_zlabel(" x (m)")    
# fig.colorbar(graph)
# fig.show()


z, t = get_z_overtime(x = 0, k = 1e6, m = 200, tstep = 0.001)
fig, ax = plt.subplots()
ax.plot(t, z,'b')
ax.set_title( "displacement (z) against time (t)")
ax.set_ylabel('z (m)')
ax.set_xlabel('t (s)');

