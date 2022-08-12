#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:00:50 2021

@author: belinda
"""

from math import *
import time
import numpy as np
from matplotlib.pyplot import *
import sim
#definition of the modified DH parameters
alpha0 = alpha2 = alpha3 = 0
alpha1 = pi/2
alpha4 = pi/2
alpha5 =-pi/2

a0 = a1 = a4 = a5 = 0
a2 = -0.612
a3 = -0.5723
a6 = 0.0922




r1 = 0.1273
r2 = 0.163941
r3 = 0
r4 = 0
r5 = 0.1157
r6 = 0

O67_h = np.array([[0],[0],[a6],[1]])

#definition of transition matrices
T01 = np.zeros((4,4))
T12 = np.zeros((4,4))
T23 = np.zeros((4,4))
T34 = np.zeros((4,4))
T45 = np.zeros((4,4))
T56 = np.zeros((4,4))
T06 = np.zeros((4,4))

def passageMatrix(dh):

    alpha = dh[0]
    a = dh[1]
    theta = dh[2]
    r = dh[3]
    return (np.array([[cos(theta), -sin(theta), 0, a], [cos(alpha)*sin(theta), cos(alpha)*cos(theta), -sin(alpha), -r*sin(alpha)], [sin(alpha)*sin(theta), sin(alpha)*cos(theta), cos(alpha), r*cos(alpha)], [0, 0, 0, 1]]))

def DKM (theta1,theta2,theta3,theta4,theta5,theta6):
    DH = np.array([[0, 0, theta1, 0.1273], [pi/2, 0, theta2, 0.163941], [0,  -0.612, theta3, 0],[0,  -0.5723, theta4, 0], [pi/2, 0, theta5, 0.1157], [-pi/2, 0, theta6, 0]])
    T01 = passageMatrix(DH[0, :])
    T12 = passageMatrix(DH[1, :])
    T23 = passageMatrix(DH[2, :])
    T34 = passageMatrix(DH[3, :])
    T45 = passageMatrix(DH[4, :])
    T56 = passageMatrix(DH[5, :])
    
    #we compute the homogeneous matrix grom R0 to Rn
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    
    O07_h = T06 @ O67_h
    O07 = O07_h[0:3]
    R = T06[0:3 , 0:3]
    alpha = atan2(R[1,0], R[0,0])
    beta  = atan2(-R[2,0],sqrt(R[0,0]**2 + R[1,0]**2))
    gamma = atan2(R[2,1], R[2,2])
    orientation = np.array([[gamma],[beta],[alpha]])
    X = np.concatenate((O07,orientation))
    return X,T02,T03,T04,T05, T06 , R


def jacobian(theta1, theta2, theta3, theta4, theta5, theta6, R ):
    r7= 0.0922
    D = np.array([[0, r7*R[2,2], -r7*R[1,2]],[-r7*R[2,2], 0, r7*Ri[0,2]],[r7*R[1,2], -r7*R[0,2],0]])
    DH = np.array([[0, 0, theta1, 0.1273], [pi/2, 0, theta2, 0.163941], [0,  -0.612, theta3, 0],[0,  -0.5723, theta4, 0], [pi/2, 0, theta5, 0.1157], [-pi/2, 0, theta6, 0]])
    
    T01 = passageMatrix(DH[0, :])
    T12 = passageMatrix(DH[1, :])
    T23 = passageMatrix(DH[2, :])
    T34 = passageMatrix(DH[3, :])
    T45 = passageMatrix(DH[4, :])
    T56 = passageMatrix(DH[5, :])
    
    
    T02 = DKM (theta1,theta2,theta3,theta4,theta5,theta6)[1]
    T03 = DKM (theta1,theta2,theta3,theta4,theta5,theta6)[2]
    T04 = DKM (theta1,theta2,theta3,theta4,theta5,theta6)[3]
    T05 = DKM (theta1,theta2,theta3,theta4,theta5,theta6)[4]
    T06 = DKM (theta1,theta2,theta3,theta4,theta5,theta6)[5]
    
    P16 = T06[0:3,3]-T01[0:3,3]
    P26 = T06[0:3,3]-T02[0:3,3]
    P36 = T06[0:3,3]-T03[0:3,3]
    P46 = T06[0:3,3]-T04[0:3,3]
    P56 = T06[0:3,3]-T05[0:3,3]
    P66 = T06[0:3,3]-T06[0:3,3]
    J11 = np.cross(T01[0:3,2],P16)
    J12 = np.cross(T02[0:3,2],P26)
    J13 = np.cross(T03[0:3,2],P36)
    J14 = np.cross(T04[0:3,2],P46)
    J15 = np.cross(T05[0:3,2],P56)
    J16 = np.cross(T06[0:3,2],P66)
    
    Z1 = T01[0:3,2]
    Z2 = T02[0:3,2]
    Z3 = T03[0:3,2]
    Z4 = T04[0:3,2]
    Z5 = T05[0:3,2]
    Z6 = T06[0:3,2]
    
    
    Jn = np.array([[J11[0], J12[0], J13[0], J14[0], J15[0], J16[0]],
                  [J11[1], J12[1], J13[1], J14[1], J15[1], J16[1]],
                  [J11[2], J12[2], J13[2], J14[2], J15[2], J16[2]],
                  [Z1[0], Z2[0], Z3[0], Z4[0], Z5[0], Z6[0]],
                  [Z1[1], Z2[1], Z3[1], Z4[1], Z5[1], Z6[1]],
                  [Z1[2], Z2[2], Z3[2], Z4[2], Z5[2], Z6[2]]])
    
    
    C = np.eye(3,3)

    I = np.eye(3,3)

    O = np.zeros((3,3))
    Conc_1 = np.concatenate((I,O))
    
    Conc_2 =  np.concatenate((D,C))
    
    M = np.concatenate((Conc_1,Conc_2),axis=1)
    
    J = M@Jn
    
    
    return J

def trajectory_generation(t,tf,xi, xf, Ri , Rf):
    
    D = xf - xi
    r = 10 * ((t/tf) ** 3) - 15 * ((t/tf)**4) + 6 * ((t/tf)**5)
    r_dot = (30 /(tf ** 3 )) * (t ** 2) - (60/(tf**4)) * (t**3) + (30/(tf**5)) * ((t**4))
    # first we compute u and v 
    R = (Ri.T) @ Rf

    c_v = (1/2) * (R[0,0] + R[1,1] + R[2,2]-1)
    s_v= (1/2) * sqrt(((R[2,1] - R[1,2])**2) + ((R[0,2] - R[2,0]) ** 2) + ((R[1,0] - R[0,1]) ** 2) )
    v = atan2(s_v,c_v)
    
    u = (1/(2*s_v)) * np.array([[R[2,1] - R[1,2]],
                                [R[0,2] - R[2,0]],
                                [R[1,0] - R[0,1]]])

    
     
    Xd = xi + r * D
    ux = u[0]
    uy = u[1]
    uz = u[2]
    rot_u_v = np.array([[(ux**2) * (1 - cos(r*v)) + cos(r*v)   ,  ux*uy*(1-cos(r*v)) - uz*sin(r*v), ux*uz*(1-cos(r*v)) + uy * sin(r*v)],
                        [ ux*uy  * (1 - cos(r*v)) + uz*sin(r*v), (uy**2) * (1-cos(v)) + cos(r*v) , uy*uz*(1-cos(r*v)) - ux * sin(r*v)],
                        [ ux*uz  * (1 - cos(r*v)) - uy*sin(r*v), uy*uz * (1-cos(r*v)) + ux*sin(r*v), (uz**2) * (1-cos(r*v)) + cos(r*v)]                 
                        ]).reshape(3,3)
    Rd = (Ri@(rot_u_v))
    X_dot = r_dot * D
    omega_d =(Ri * r_dot * v) @ u

    return Xd, Rd, X_dot, omega_d


if __name__ == "__main__":
    
    print ('Program started')
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
    if clientID!=-1:
        print ('Connected to remote API server')

        startTime=time.time()
        sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
                   
        h = np.array([0,0,0,0,0,0])

        r, h[0]=sim.simxGetObjectHandle(clientID,'UR10_joint1', sim.simx_opmode_blocking)
        r, h[1]=sim.simxGetObjectHandle(clientID,'UR10_joint2', sim.simx_opmode_blocking)
        r, h[2]=sim.simxGetObjectHandle(clientID,'UR10_joint3', sim.simx_opmode_blocking)
        r, h[3]=sim.simxGetObjectHandle(clientID,'UR10_joint4', sim.simx_opmode_blocking)
        r, h[4]=sim.simxGetObjectHandle(clientID,'UR10_joint5', sim.simx_opmode_blocking)
        r, h[5]=sim.simxGetObjectHandle(clientID,'UR10_joint6', sim.simx_opmode_blocking)
        
        

         

        theta1 = np.deg2rad(0)
        theta2 = np.deg2rad(0)
        theta3 = np.deg2rad(0)
        theta4 = np.deg2rad(0)
        theta5 = np.deg2rad(0)
        theta6 = np.deg2rad(0)

        

            
        sim.simxSetJointTargetPosition(clientID, h[0], theta1, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(clientID, h[1], theta2, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(clientID, h[2], theta3, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(clientID, h[3], theta4, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(clientID, h[4], theta5, sim.simx_opmode_blocking)
        sim.simxSetJointTargetPosition(clientID, h[5], theta6, sim.simx_opmode_blocking)
        
        #time.sleep(4)
       
        r, theta1=sim.simxGetJointPosition(clientID, h[0], sim.simx_opmode_blocking)
        r, theta2=sim.simxGetJointPosition(clientID, h[1], sim.simx_opmode_blocking)
        r, theta3=sim.simxGetJointPosition(clientID, h[2], sim.simx_opmode_blocking)
        r, theta4=sim.simxGetJointPosition(clientID, h[3], sim.simx_opmode_blocking)
        r, theta5=sim.simxGetJointPosition(clientID, h[4], sim.simx_opmode_blocking)
        r, theta6=sim.simxGetJointPosition(clientID, h[5], sim.simx_opmode_blocking)
        
        Xi = DKM(theta1, theta2, theta3, theta4, theta5, theta6 )[0]
        xi = Xi[:3,0]
        Ri = DKM(theta1, theta2, theta3, theta4, theta5, theta6 )[6]
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Number of objects in the scene: ', len(objs))
        else:
            print('Remote API function call returned with error code: ', res)

        startTime = time.time()
        sim.simxGetIntegerParameter(clientID, sim.sim_intparam_mouse_x, sim.simx_opmode_streaming)
        
        theta1f= np.deg2rad(1.4)
        theta2f = np.deg2rad(-70)
        theta3f = np.deg2rad(50.65)
        theta4f = np.deg2rad(-45.75)
        theta5f = np.deg2rad(12.35)
        theta6f = np.deg2rad(-10.07)
        
        Rf = DKM(theta1f, theta2f, theta3f, theta4f, theta5f, theta6f )[6]
        Xf = DKM(theta1f, theta2f, theta3f, theta4f, theta5f, theta6f )[0]
        xf = Xf[:3,0]
        
        D = xf - xi
        N= 2000
        tf =10
        t = np.zeros((1, N))
        kp = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
        eo =np.zeros((3,1))
        Xd_data_1 = []
        Xd_data_2 = []
        Xd_data_3 = []
        
        Xa_data_1 = []
        Xa_data_2 = []
        Xa_data_3 = []       
        temps = []

        eo_plot1 = []
        eo_plot2= []
        eo_plot3 = [] 
        dt =0.1
        K0 = np.array([[5,0,0],[0,5,0],[0,0,5]])
        to= time.time()
        i = 0

        theta1a = theta1
        theta2a = theta2
        theta3a = theta3
        theta4a = theta4
        theta5a = theta5
        theta6a = theta6
        
        J = jacobian(theta1, theta2, theta3, theta4, theta5, theta6, Ri)
        
#         while (time.time()-to)<tf:
#         ##if False:
#               t=time.time()-to
              
              

#               #Desired X and R
#               xd = trajectory_generation(t,tf,xi, xf, Ri , Rf)[0]
              
#               Rd = trajectory_generation(t,tf,xi, xf, Ri , Rf)[1]
              
#               # position actuelle:
#               r, theta1a=sim.simxGetJointPosition(clientID, h[0], sim.simx_opmode_blocking)
#               r, theta2a=sim.simxGetJointPosition(clientID, h[1], sim.simx_opmode_blocking)
#               r, theta3a=sim.simxGetJointPosition(clientID, h[2], sim.simx_opmode_blocking)
#               r, theta4a=sim.simxGetJointPosition(clientID, h[3], sim.simx_opmode_blocking)
#               r, theta5a=sim.simxGetJointPosition(clientID, h[4], sim.simx_opmode_blocking)
#               r, theta6a=sim.simxGetJointPosition(clientID, h[5], sim.simx_opmode_blocking)
        
#               theta1_pre = theta1a
#               theta2_pre = theta2a
#               theta3_pre = theta3a
#               theta4_pre = theta4a
#               theta5_pre = theta5a
#               theta6_pre = theta6a
              
#               Xa = DKM(theta1a, theta2a, theta3a, theta4a, theta5a, theta6a)[0]
              
#               xa = Xa[:3,0]
#               Ra = DKM(theta1a, theta2a, theta3a, theta4a, theta5a, theta6a)[6]
              
#               #Erreur en position
#               ep = xd - xa
              
#               #
#               # Xpoit desirée   
#               Xd_dot = trajectory_generation(t,tf,xi, xf, Ri , Rf)[2]
              
#               # multiplication par un gain Kp

#               A1 = Xd_dot + np.dot(kp, ep)
              
              
              
#               nd = Rd[:,0]
#               sd = Rd[:,1]
#               ad = Rd[:,2]
     
#               # Re That corresponds to the actual orientation
#               ne = Ra[:,0]
#               se = Ra[:,1]
#               ae = Ra[:,2]
              
#               S_nd = np.array([[ 0  , -nd[2],  nd[1]],
#                             [ nd[2], 0     , -nd[0]],
#                             [-nd[1], nd[0] ,     0]])
            
#               S_sd = np.array([[ 0  , -sd[2],  sd[1]],
#                             [ sd[2], 0     , -sd[0]],
#                             [-sd[1], sd[0] ,     0]])
            
#               S_ad = np.array([[ 0  , -ad[2],  ad[1]],
#                             [ ad[2], 0     , -ad[0]],
#                             [-ad[1], ad[0] ,     0]])
            
#               S_ne = np.array([[ 0  , -ne[2],  ne[1]],
#                             [ ne[2], 0     , -ne[0]],
#                             [-ne[1], ne[0] ,     0]])
            
#               S_se = np.array([[ 0  , -se[2],  se[1]],
#                             [ se[2], 0     , -se[0]],
#                             [-se[1], se[0] ,     0]])
            
#               S_ae = np.array([[ 0  , -ae[2],  ae[1]],
#                             [ ae[2], 0     , -ae[0]],
#                             [-ae[1], ae[0] ,     0]])
        
#               E0= 0.5* (np.cross(Ra[:,0], Rd[:,0]) + np.cross(Ra[:,1], Rd[:,1]) + np.cross(Ra[:,2], Rd[:,2]))
#               eo[:,0] =E0
#               L = -0.5 * (np.dot(S_nd, S_ne) + np.dot(S_sd, S_se) + np.dot(S_ad, S_ae))
#               #L =np.asmatrix(L0)
#               L_pinv=np.linalg.pinv(L)
             
#               #vitesse angulaire
#               wd = trajectory_generation(t,tf,xi, xf, Ri , Rf)[3]
              
#               # Jacob
#               J = jacobian(theta1a, theta2a, theta3a, theta4a, theta5a, theta6a, Ra)

#               # Pseudo inverse
#               J_pinv = np.linalg.pinv(J)
              
#               # Desired Velocity
              
#               LTW =np.dot(L.T ,wd)
#               kpp = np.dot(K0,eo)
#               XXX = (LTW + kpp).reshape(3,1)
                       
#               yyy =np.dot(L_pinv, XXX)

#               F = np.array([[A1[0]],
#                             [A1[1]],
#                             [A1[2]],
#                             [yyy[0]],
#                             [yyy[1]],
#                             [yyy[2]]])
  
#               qd_point = np.dot(J_pinv, F).reshape(6,1)
              
#               temps.append(t)
             
#               Xd_data_1.append(xd[0])
#               Xd_data_2.append(xd[1])
#               Xd_data_3.append(xd[2])

#               Xa_data_1.append(xa[0])
#               Xa_data_2.append(xa[1])
#               Xa_data_3.append(xa[2])              

#               eo_plot1.append(eo[0])
#               eo_plot2.append(eo[1])
#               eo_plot3.append(eo[2])


#               i = i +1

#               # Intégration
#               qd_0 = qd_point[0]*dt + theta1a
#               qd_1 = qd_point[1]*dt + theta2a
#               qd_2 = qd_point[2]*dt + theta3a
#               qd_3 = qd_point[3]*dt + theta4a
#               qd_4 = qd_point[4]*dt + theta5a
#               qd_5 = qd_point[5]*dt + theta6a

#               # Servomotor
#               sim.simxSetJointTargetPosition(clientID, h[0], qd_0, sim.simx_opmode_blocking)
#               sim.simxSetJointTargetPosition(clientID, h[1], qd_1, sim.simx_opmode_blocking)
#               sim.simxSetJointTargetPosition(clientID, h[2], qd_2, sim.simx_opmode_blocking)
#               sim.simxSetJointTargetPosition(clientID, h[3], qd_3, sim.simx_opmode_blocking)
#               sim.simxSetJointTargetPosition(clientID, h[4], qd_4, sim.simx_opmode_blocking)
#               sim.simxSetJointTargetPosition(clientID, h[5], qd_5, sim.simx_opmode_blocking)


# sim.simxFinish(clientID)
    
# fig, axs = subplots(3,1) 

# axs[0].plot(temps , Xd_data_1)
# axs[0 ].plot(temps, Xa_data_1, 'r')
# axs[0].set_title('desired position vs real positio')
# axs[1 ].plot(temps , Xd_data_2)
# axs[1 ].plot(temps, Xa_data_2,'r')
# axs[2 ].plot(temps , Xd_data_3)
# axs[2 ].plot(temps, Xa_data_3,'r')


# Fig , axes= subplots(3,1)
# axes[0].plot(temps, eo_plot1)
# axes[0].set_title('orientation error')
# axes[1].plot(temps, eo_plot2)
# axes[2].plot(temps, eo_plot3)

       


        
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    

    
    
    
    
    
    

    
    