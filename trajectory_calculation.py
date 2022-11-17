#!/usr/bin/env python
# coding: utf-8
# created by aashishgupta3008@gmail.com on 2nd Nov. 2022

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import linalg as LA
from astropy import units as u
from astropy import constants as const

def falling_trajectory(x0,y0,z0,vx0,vy0,vz0,Ms_val,verbose = True,ang_range = 0.5*np.pi):
    
    '''
    Returns positions and velocities for an infalling particle based on Mendoza+09
    
    Args:
    x0: Initial position in R.A. [au]
    y0: Initial position in Decl. [au]
    z0: Initial position in radial distance [au]
    vx0: Initial velocity in R.A. [km/s]
    vy0: Initial velocity in Decl. [km/s]
    vz0: Initial velocity in radial distance [km/s]
    Ms_val: Mass of central object(star) [solar mass]
    verbose: Set False to not print intermediate outputs, default is True 
    ang_range: Range of (parametric) angles to compute the trajectory upto, 2*pi will give you complete trajectory but it will be weird to plot in case of parabola/hyperbola. [radian]
    
    Output:
    c1_t: Particle's positions along the trajectory in original coordinates. Each sub-array is one axis.
    vc1_t: Particle's velocities along the trajectory in original coordinates. Each sub-array is one axis.
    sr: Particle's positions along the trajectory in spherical coordinates. Each sub-array is one axis.
    sv: Particle's velocities along the trajectory in spherical coordinates. Each sub-array is one axis.
    '''

    Ms = Ms_val*u.solMass
    r0_v = np.array([x0,y0,z0])#*u.au
    v0_v = np.array([vx0,vy0,vz0])#*u.km/u.s

    ### Unit vectors of rotated frame
    r0_mag = LA.norm(r0_v)
    r0_e = r0_v/r0_mag
    r0_e

    n_v = np.cross(r0_v,v0_v)  #normal to r0_v and v0_v plane
    n_mag = LA.norm(n_v)
    n_e = n_v/n_mag
    # n_e

    m_v = np.cross(r0_v,n_v)  #normal to r0_v and v0_v plane
    m_mag = LA.norm(m_v)
    m_e = m_v/m_mag
    # m_e

    ### Coordinate transformation
    trans1 = np.array([m_e,r0_e,n_e])
    inv_trans1 = LA.inv(trans1)  # inverse transformation matrix
    c1 = r0_v#[1000,-2321,0.01]    # coordinate in original ra, dec, dist..
    c2 = np.matmul(trans1,c1)     # coord. in new m, r0, n...
    if verbose:
        print("Initial coords in rotated cartesian frame:",c2)

    r0 = np.sqrt(sum(c2**2))
    theta0 = np.arccos(c2[2]/r0)  # should always for pi/2
    phi0 = np.arctan2(c2[1],c2[0]) # should be pi/2 for r0
    sr0 = np.array([r0, theta0, phi0])
    if verbose:
        print("Initial coords in spherical frame:",sr0)

    # going back to c2 coords
    x2f = r0*np.sin(theta0)*np.cos(phi0)
    y2f = r0*np.sin(theta0)*np.sin(phi0)
    z2f = r0*np.cos(theta0)
    c2f = [x2f,y2f,z2f]
    # np.isclose(c2,[x2f,y2f,z2f]).all() # should be True
    # going back to c1 coords
    f = np.matmul(inv_trans1,c2f)
    if verbose:
        print("Can we get back the original coords?:",np.isclose(f,c1).all()) # should be True

    v0_v2 = np.matmul(trans1,v0_v) 
    if verbose:
        print("Initial vel. in rotated cartesian frame:",v0_v2)
    trans2 = np.array([[np.sin(theta0)*np.cos(phi0),np.sin(theta0)*np.sin(phi0),np.cos(theta0)],
                       [np.cos(theta0)*np.cos(phi0),np.cos(theta0)*np.sin(phi0),-np.sin(theta0)],
                       [-np.sin(phi0),np.cos(phi0),0]])
    sv0 = np.matmul(trans2,v0_v2)
    if verbose:
        print("Initial vel. in spherical frame:",sv0)

    r0 = r0*u.au  # give units
    vr0 = sv0[0]*u.km/u.s  # initial vel. in r
    vphi0 = sv0[2]*u.km/u.s   # initial vel. in phi
    h0 = vphi0*r0   # initial (conserved) angular momentum w.r.t. azimuth (assuming theta0 = pi/2)

    ### Calculate total energy and other constants
    E = (0.5*(vr0**2)) + (0.5*((h0*np.sin(theta0)/r0)**2)) - (const.G*Ms/r0)  # Eq. 2 in Mendoza+09
    if verbose:
        print("Total energy:",E)

    ru = (h0**2)/(const.G*Ms)
    E0 = (const.G*Ms)/ru
    vk = np.sqrt(E0)

    mu = np.sqrt((ru/r0)**2)  # Eq. 3 in Mendoza+09 (also np.sqrt(((h0/r0)**2)/E0))
    nu = np.sqrt((vr0**2)/E0)  # Eq. 3 in Mendoza+09
    # mu,nu

    epsilon = 2*E/E0
    ecc = np.sqrt(1+epsilon*(np.sin(theta0)**2))   ## NEED TO CHECK
    if verbose:
        print("Orbit eccentricity:",ecc)

    ### Computing trajectory
    pang0 = np.arccos((1/ecc)*(1-mu*(np.sin(theta0)**2))).to(u.rad).value  # parametric angle, small phi (eq. 7) in Mendoza+09
    d_pangs = np.arange(0,ang_range,np.pi/50) 
    pang = pang0 -np.sign(vr0)*d_pangs
    # pang = pang0 + d_pangs
    # pang = np.arange(pang0,1.5*np.pi,np.pi/50)  

    r = ((np.sin(theta0)**2)/(1-ecc*np.cos(pang)) * ru).to(u.au)  # eq. 5 in Mendoza+09
    theta = np.array([np.pi/2]*len(pang))
    # phi = ((np.pi*3/2 - (pang-pang0))%(2*np.pi) - np.pi)  # assumed theta = 90, added %2 because phi is between -pi to +pi
    phi = ((np.pi*3/2 + np.sign(vphi0)*d_pangs)%(2*np.pi) - np.pi)  # assumed theta = 90, added %2 because phi is between -pi to +pi
    sr = [r.value,theta,phi]

    # plt.plot((pang-pang0)/np.pi,r.value)
    # plt.show()

    vphi = ((np.sin(theta0)**2)/((r/ru)*np.sin(theta))*vk).to(u.km/u.s)  # eq. 11 in Mendoza+09
    vtheta = (((np.sin(theta0)/((r/ru)*np.sin(theta)))*np.sqrt((np.cos(theta0)**2)-(np.cos(theta)**2)))*vk).to(u.km/u.s) # eq. 12 in Mendoza+09
    # vr = (((-1*ecc*np.sin(pang)*np.sin(theta0))/((r/ru)*(1-ecc*np.cos(pang))))*vk).to(u.km/u.s)  # eq. 13 in Mendoza+09
    vr = (((np.sign(vr0)*ecc*np.sin(pang)*np.sin(theta0))/((r/ru)*(1-ecc*np.cos(pang))))*vk).to(u.km/u.s)  # eq. 13 in Mendoza+09
    sv = np.array([vr,vtheta,vphi])

    ### Converting trajectory to orginial coords
    x2_t = r*np.sin(theta)*np.cos(phi)
    y2_t = r*np.sin(theta)*np.sin(phi)
    z2_t = r*np.cos(theta)
    c2_t = [x2_t,y2_t,z2_t]
    c1_t = np.matmul(inv_trans1,c2_t) # going back to c1 coords

    # converting velocites
    vc2_t = []
    for i in range(len(r)):
    # for i in range(10):
        trans2_gen = np.array([[np.sin(theta[i])*np.cos(phi[i]),np.sin(theta[i])*np.sin(phi[i]),np.cos(theta[i])],   #General transformation matrix to c2
                               [np.cos(theta[i])*np.cos(phi[i]),np.cos(theta[i])*np.sin(phi[i]),-np.sin(theta[i])],
                               [-np.sin(phi[i]),np.cos(phi[i]),0]])
        inv_trans2_gen = LA.inv(trans2_gen)
        vc2_t += [np.matmul(inv_trans2_gen,sv.T[i]) ]
    vc2_t = np.array(vc2_t).T
    vc1_t = np.matmul(inv_trans1,vc2_t)  # Vel. tranformation from C2 to C1

    
    ### Checks before returning
    check1 = np.isclose(c1_t.T[0],r0_v).all()  # Trajectory starts from correct position
    if ~check1:
        print("Initial positions don't match.")

    check2 = np.isclose(vc1_t.T[0],v0_v).all()  # Trajectory starts from correct velocity
    if ~check2:
        print("Initial velocitied don't match.")

    # checking signs of gradients
    for i in range(len(c1_t)):
        x = c1_t[i]
        v = vc1_t[i]
        if ~(np.sign(np.gradient(x)) == np.sign(v+1e-4)).all():
            print("Sign of position gradients do not match velocities.")

    return(c1_t,vc1_t,sr,sv)


