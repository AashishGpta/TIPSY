#!/usr/bin/env python
# coding: utf-8
# created by aashishgupta3008@gmail.com on 2nd Nov. 2022

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
from astropy import units as u
from astropy import constants as const
import plotly.graph_objects as go
from sklearn.model_selection import ParameterGrid
from scipy.interpolate import interp1d
import time

def arcsec2au(a,dist):  
    ''' take values in arcsec, return in au using distance (with units) '''
    a2 = (a*u.arcsec).to(u.radian).value
    return((a2*dist).to(u.au).value)

def au2arcsec(a,dist):  
    ''' take values in au, return in arcsec using distance (with units) '''
    a2 = ((a*u.au/dist.to(u.au))*u.radian)
    return(a2.to(u.arcsec).value)

def spherical_coords(x,y):
    ''' Converts cartesian coordinates (x,y) to polar coordinates.'''
    r = np.sqrt((x**2)+(y**2)) # total distance from center
    theta = np.arctan2(y,x)  # Angle wrt x-axis, in radians
    return(r,theta)

def falling_trajectory(x0,y0,z0,vx0,vy0,vz0,Ms_val,verbose = True,ang_range = 0.5*np.pi,ang_step = np.pi/50):
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
    ang_step: Resolution of solutions in terms of (parametric) angles. [radian]
    
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

    n_v = np.cross(r0_v,v0_v)  #normal to r0_v and v0_v plane
    n_mag = LA.norm(n_v)
    n_e = n_v/n_mag

    m_v = np.cross(r0_v,n_v)  #normal to r0_v and v0_v plane
    m_mag = LA.norm(m_v)
    m_e = m_v/m_mag

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
    # nu = np.sqrt((vr0**2)/E0)  # Eq. 3 in Mendoza+09
    # mu,nu

    epsilon = 2*E/E0
    ecc = np.sqrt(1+epsilon*(np.sin(theta0)**2))   ## NEED TO CHECK
    if verbose:
        print("Orbit eccentricity:",ecc)

    ### Computing trajectory
    pang0 = np.arccos(np.round((1/ecc)*(1-mu*(np.sin(theta0)**2)),6)).to(u.rad).value  # parametric angle, small phi (eq. 7) in Mendoza+09, np.round to round off no. slightly greater than 1
#     pang0 = np.arccos((1/ecc)*(1-mu*(np.sin(theta0)**2))).to(u.rad).value  # parametric angle, small phi (eq. 7) in Mendoza+09
    d_pangs = np.arange(0,ang_range,ang_step) 
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
    check1 = np.isclose(c1_t.T[0],r0_v,rtol=1e-2,atol=1e-2).all()  # Trajectory starts from correct position
    if ~check1:
        print("Potetial Inconsistency: Given initial positions:",r0_v,"Calculated initial positions:",c1_t.T[0])

    check2 = np.isclose(vc1_t.T[0],v0_v,rtol=1e-2,atol=1e-2).all()  # Trajectory starts from correct velocity
    if ~check2:
        print("Potetial Inconsistency: Given initial velocities:",v0_v,"Calculated initial velocities:",vc1_t.T[0])
    
    dx = r[1:]-r[:-1]
    dphi = (phi[1:]-phi[:-1])%(2*np.pi)
    r_avg = (r[1:]+r[:-1])/2
    v1_avg = (vr[1:]+vr[:-1])/2
    v2_avg = (vphi[1:]+vphi[:-1])/2
    dt = (dx/v1_avg).to(u.yr)
    # time = np.cumsum(dt)
    v2_avg2 = (r_avg*dphi/dt).to(vphi.unit)
    check3 = (abs((v2_avg2-v2_avg)/v2_avg)[1:-1] < 0.2).all()#<0.0000001*v2_avg
    if ~check3:
        print("Cannot reporduce velocities using gradients.")

#     # checking signs of gradients
#     for i in range(len(c1_t)):
#         x = c1_t[i]
#         v = vc1_t[i]
#         if ~(np.sign(np.gradient(x)) == np.sign(v+1e-4)).all():
#             print("Sign of position gradients do not match velocities.")

    return(c1_t,vc1_t,sr,sv)

def rebound_trajectory(x0,y0,z0,vx0,vy0,vz0,Ms_val,t_max = 1e4,verbose = True):
    '''
    Returns positions and velocities for an infalling particle based on rebound simulations. Can be used for comparison (or as an alternative to trajectories from Mendoza et al. 2009 models.
    
    Args:
    x0: Initial position in R.A. [au]
    y0: Initial position in Decl. [au]
    z0: Initial position in radial distance [au]
    vx0: Initial velocity in R.A. [km/s]
    vy0: Initial velocity in Decl. [km/s]
    vz0: Initial velocity in radial distance [km/s]
    Ms_val: Mass of central object(star) [solar mass]
    t_max: Time (in years) to compute the trajectory upto
    verbose: Set False to not print intermediate outputs, default is True 
    
    Output:
    xyz: Particle's positions along the trajectory in original coordinates. Each sub-array is one axis.
    v_xyz: Particle's velocities along the trajectory in original coordinates. Each sub-array is one axis.
    '''
    import rebound
    
    N_dim = 3    # Total dimensions of problem, 3 for x/y/z
    N_particle = 2    # Total no. of particles in simulation

    vx0_auyr = (vx0*(u.km/u.s)).to((u.au/u.yr)).value   # Velocity in au/yr
    vy0_auyr = (vy0*(u.km/u.s)).to((u.au/u.yr)).value
    vz0_auyr = (vz0*(u.km/u.s)).to((u.au/u.yr)).value

    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.add(m=Ms_val)  # Add central mass
    # sim.add(m=1.,a=0.5)  # Add central mass
    sim.move_to_com()  
    sim.integrator = "whfast"
    sim.dt = 0.005
    sim.add(x=x0,y=y0,z=z0,vx=vx0_auyr,vy=vy0_auyr,vz=vz0_auyr) # mass is set to 0 by default, random true anomaly   

    if verbose:
        sim.status()
    # sim.N_active = 2

    N_out = 100
    xyz = np.zeros((N_out, N_particle, N_dim))
    v_xyz = np.zeros((N_out, N_particle, N_dim))
    times = np.linspace(0, t_max, N_out)
    for i, time1 in enumerate(times):
        sim.integrate(time1)
        for j, p in enumerate(sim.particles[:]):
            xyz[i][j] = [p.x, p.y,p.z]        
            v_xyz[i][j] = [p.vx, p.vy,p.vz]
    v_xyz = (v_xyz*u.au/u.yr).to(u.km/u.s).value
    return(xyz,v_xyz)
    
def traj_fitting(streamer_cube,Ms_val,dist,svel,N_elements=10,theta_weight=1
                 ,vxy_ang0=None,vxy_ang0_span=None,vxy_ang0_step=0.0872
                 ,z0_lim=None,z0_step=None,vxy0_lim=None,vxy0_step=None
                 ,show_dist_plots=True,show_vel_ang=True,show_fit_cost=True,show_fit_3d=True,fit2html=False,html_path=None
                 ,verbose=True):
    '''
    Code to fit infalling trajectories in PPV space to streamers.
    
    Args:
    streamer_cube: SpectralCube object with only streamer emission
    Ms_val ([float]): Mass of central object [solar mass]
    dist ([float]): Distance to the source [parsecs]
    svel ([float]): Systematic radial velocity of the source [km/s]
    N_elements (Optional[int]): No. of elements to use for fitting
    theta_weight (Optional[int]): Weightage of projected angles along the streamer for distance metric, 0 means just using projected distances
    vxy_ang0 (Optional[float]): Initial angle between P.O.S. velocity and R.A. axis [radians]
    vxy_ang0_span (Optional[float]): Total span of initial angles between P.O.S. velocity and R.A. axis to use for parameter space [radians]
    vxy_ang0_step (Optional[float]): Resolution of initial angles between P.O.S. velocity and R.A. axis to use for parameter space, default is ~5 degrees [radians]
    z0_lim (Optional[tuple]): Min. and max. value of z0 (rel. radial dist.) to generate trajectories [au]
    z0_step (Optional[float]): Resolution for z0 to generate trajectories [au]
    vxy0_lim (Optional[tuple]): Min. and max. value of vxy0 (speed in RA/Decl. plane) to generate trajectories [km/s]
    vxy0_step (Optional[float]): Resolution for z0 to generate trajectories [km/s]    
    show_dist_plots (Optional[bool]): Whether to show plots of projected distances, angles, and final distance metrics (independent parameter for fitting)
    show_vel_ang (Optional[bool]): Whether to show estimation of initial orientation of P.O.S. velocity
    show_fit_cost (Optional[bool]): Whether to show plot with fitting fraction (and deviation) as a function of free parameters
    show_fit_3d (Optional[bool]): Whether to show 3D plot original point cloud, points used for fitting and final trajectory 
    fit2html (Optional[bool]): Whether to save 3D fitting plot to .html file
    html_path (Optional[str]): Path to save 3D fitting plot to .html file
    verbose (Optional[bool]): Whether to display miscellaneous text outputs 
    
    Returns:
    param_grid2: Pandas DataFrame with all the parameter combinations used and corresponding deviations
    traj_m: Trajectory corresponding to the best fit
    [traj_comp_m,pcmeans,pcstds]: List of three 3xN_elements numpy arrays storing values for interpolated traj_m, means of streamer, and standard deviations of streamer
'''
    ## Cube to point cloud    
    pcloud = np.array(streamer_cube)
    rms_mask = ~np.isnan(pcloud)
    flux = pcloud[rms_mask]
    pcinds = np.indices(pcloud.shape)#.shape   #Array of indices of all points in streamer point cloud
    pcz = pcinds[0][rms_mask]    #Individual indices for non nan values (high enough SNR)
    pcy = pcinds[1][rms_mask]
    pcx = pcinds[2][rms_mask]

    pccoords = np.array(streamer_cube.wcs.pixel_to_world_values(np.array([pcx,pcy,pcz]).T)).T   # Indices to FITS coords
    pccoords[0] = (pccoords[0]-streamer_cube.header['CRVAL1'])*60*60   # Coords in units of PPV diagram, R.A.
    pccoords[1] = (pccoords[1]-streamer_cube.header['CRVAL2'])*60*60   # Decl.
    pccoords[2] = (pccoords[2]-streamer_cube.header['CRVAL3'])         # Radial Vel.

    ## Distance metric (independent parameter)
    pcd_r,pcd_theta = spherical_coords(pccoords[0],pccoords[1])
    r_percentile_thresh = 100/N_elements  # No. of points to consider for ref. angle
    theta_ref = np.median(pcd_theta[pcd_r<np.percentile(pcd_r,r_percentile_thresh)])
    # now we are interested in deviation of pcd_theta wrt theta_ref and this is not straightforward
    # we do not want deviation for angles close to theta_ref to be close to +-360
    # one way around is to have cyclic deviation upto 180, this is implemented below
    # IMP: this method will fail if the deviations >180 needs to be accounted in distance, i.e., if spirals are very closely wound
    pcd_theta2 = np.pi-np.abs(np.pi-np.abs(pcd_theta-theta_ref))
    pcd = pcd_r*np.sqrt(1+(theta_weight*pcd_theta2)**2)

    
    if show_dist_plots:
#         plt.scatter(pccoords[0],pccoords[1],c=pcd_r)
#         plt.colorbar(label='Projected distance (r)')
#         plt.xlabel('R.A. [arcsec]')
#         plt.ylabel('Decl. [arcsec]')
#         plt.show()

#         plt.scatter(pccoords[0],pccoords[1],c=pcd_theta2*180/np.pi)
#         plt.colorbar(label=r'Projected angle ($\theta$)')
#         plt.xlabel('R.A. [arcsec]')
#         plt.ylabel('Decl. [arcsec]')
#         plt.show()

        plt.scatter(pccoords[0],pccoords[1],c=pcd)
        plt.colorbar(label='Distance metric')
        plt.xlabel('R.A. [arcsec]')
        plt.ylabel('Decl. [arcsec]')
        plt.show()

    b_per = np.linspace(0,100,N_elements+1)  # Array of percetile values to break the array into
    pars = np.array([np.percentile(pcd,per) for per in b_per])   # pcd values at percentile values
    if verbose:
        print("Partition boundaries for projected distances:",np.round(pars,3))

    ## Point cloud to array of weighted-mean values (a curve-like representation)
    pcmeans = np.zeros((3,N_elements))  # intialize "empty" array
    pcstds = np.zeros((3,N_elements))  # intialize "empty" array
    for i in range(N_elements):
        dinds = (pcd>pars[i]) & (pcd<=pars[i+1])  # Identify points in given distance range
        pcmeans[:,i] = np.average(pccoords.T[dinds],axis=0,weights=flux[dinds])  # Add means    
        pcstds[:,i] = np.sqrt(np.average((pccoords.T[dinds]-pcmeans[:,i])**2,axis=0,weights=flux[dinds])) # Add standard deviations
    r_means,theta_means = spherical_coords(pcmeans[0],pcmeans[1])
    theta2_means = np.pi-np.abs(np.pi-np.abs(theta_means-theta_ref))  # Using same theta_ref as observations, right thing to do?
    d_means = r_means*np.sqrt(1+(theta_weight*theta2_means)**2)

    ## Initial distance (arcseec to au)
    dist = dist*u.pc # just for consistency with other codes
    x0_as = pcmeans[0][-1] #*u.arcsec
    y0_as = pcmeans[1][-1] #*u.arcsec
    x0 = arcsec2au(x0_as,dist)
    y0 = arcsec2au(y0_as,dist) #(x0_as.to(u.radian).value*dist).to(u.au).value
    ## Initial velocity (km/s)
    vz0 = pcmeans[2][-1]*1e-3-svel
    print("Initial x0, y0 and vz0:",x0,y0,vz0)


    ### Setting grid for free parameters
    ## Initial offset in radial distance
    if z0_step == None:
        try:
            bsize = streamer_cube.header['BMAJ']   # Beam size (Major axis)
        except KeyError:
            print("BMAJ field doesn't exist in fits header, trying to use pixel size (CDELT2)")
            bsize = np.abs(streamer_cube.header['CDELT2'])
        bsize_au = arcsec2au(bsize*60*60,dist)  # Beam size in au
        z0_step = np.abs(bsize_au)
    if z0_lim == None:
        dev = (np.abs(x0)+np.abs(y0))/2   # Typical deviation in au
        dev_z = (-1*np.sign(vz0))*dev   # Expected deviation in z (au), (NOTE: vz0*timescale could also be used)
        d_dev_z = max(10*z0_step,1500)  #Range of values to test
        z0_min=dev_z-d_dev_z
        z0_max=dev_z+d_dev_z
    else:
        z0_min=z0_lim[0]
        z0_max=z0_lim[1]
    z0_range = np.arange(z0_min,z0_max,z0_step)
    if verbose:
        print("Min., max. and step for initial radial sep. (z0):",z0_min,z0_max,z0_step)

    ## Initial offset in P.O.S speed
    if vxy0_step == None:
        cwidth = streamer_cube.header['CDELT3']
        vxy0_step = np.abs(cwidth)
    if vxy0_lim == None:
        dev_vxy = np.abs(vz0)*(2**0.5)    #Rough estimate of vxy0 (NOTE: dev/timescale could also be used)
        d_dev_vxy = max(10*vxy0_step,2)      # No good reason to choose 10 or 5 :P
        vxy0_min=max(0,dev_vxy-d_dev_vxy)  #vxy0_min always should be positive
        vxy0_max=dev_vxy+d_dev_vxy
    else:
        vxy0_min=vxy0_lim[0]
        vxy0_max=vxy0_lim[1]
    vxy0_range = np.arange(vxy0_min,vxy0_max,vxy0_step)
    if verbose:
        print("Min., max. and step for initial speed in P.O.S. (vxy0):",vxy0_min,vxy0_max,vxy0_step)

    ## Orientation of P.O.S velocity wrt R.A. axis
    if vxy_ang0 == None:
        dx = pcmeans[0][-2]-pcmeans[0][-1]    # Initial change in R.A.
        dy = pcmeans[1][-2]-pcmeans[1][-1]    # Initial change in Decl.
        vxy_ang0_est = np.arctan2(dy,dx) #Angle between total x-y velocity and velocity in x (R.A.)
        if vxy_ang0_span == None:    # If span for intial orientation angles is not provided, estimate it using error in estimated angle
            e_x = pcstds[0][-2]  # Error in dx
            e_y = pcstds[1][-2]  # Error in dy
            r = dy/dx
            e_r = r*np.sqrt((e_x/dx)**2 + (e_y/dy)**2)
            vxy_ang0_error = np.abs(e_r/(1+r**2)) # Error in vxy_ang0
            vxy_ang0_span = 2*vxy_ang0_error
        r1 = np.arange(vxy_ang0_est,vxy_ang0_est+(vxy_ang0_span/2),vxy_ang0_step)
        r2 = np.arange(vxy_ang0_est,vxy_ang0_est-(vxy_ang0_span/2),-vxy_ang0_step)
        vxy_ang0_range = np.concatenate([r2[::-1][:-1],r1])
    else:
        vxy_ang0_range = np.array([vxy_ang0])
    if verbose:
        print("Min., max. and step for initial directions in P.O.S. (vxy_ang0):",vxy_ang0_range[0],vxy_ang0_range[-1],vxy_ang0_step)
    if show_vel_ang:
        i1 = 0
        i2 = 1
        labels = ['R.A. offset [arcsec]','Decl. offset [arcsec]','Radial Vel. offset']  # Just put an array to make it easier in future
        plt.plot(pccoords[i1],pccoords[i2],'.',alpha = 0.1,label = 'Observations',zorder = 0)
        plt.errorbar(x = pcmeans[i1],y = pcmeans[i2],xerr = pcstds[i1],yerr = pcstds[i2]
                     ,fmt = 's-',alpha = 1, label = 'Means')
        plt.plot(x0_as,y0_as,'ko',alpha = 1)
    #     plt.plot(x_smoothed,y_smoothed,'P-',label='Spline fit')
        plt.quiver(pcmeans[0][-1],pcmeans[1][-1],np.cos(vxy_ang0_est),np.sin(vxy_ang0_est),
                   angles='xy',scale_units = 'dots',scale=0.01,label='Initial direction',zorder=5)
        for vxy_ang0_i in vxy_ang0_range:
            plt.quiver(pcmeans[0][-1],pcmeans[1][-1],np.cos(vxy_ang0_i),np.sin(vxy_ang0_i),
                       angles='xy',scale_units = 'dots',scale=0.01,zorder=5,alpha=0.3)
        plt.xlabel(labels[i1])
        plt.ylabel(labels[i2])
        plt.legend()
        plt.show()

    ## Creating parameter grid
    param_grid = {'x0':[x0],'y0':[y0],'z0':z0_range
                  ,'vxy_ang0':vxy_ang0_range,'vxy0':vxy0_range,'vz0':[vz0]
                  ,'Ms_val':[Ms_val]}
    param_grid2 = pd.DataFrame(ParameterGrid(param_grid))
    param_grid2['fit_fraction'] = np.nan
    param_grid2['deviation'] = np.nan
    N_params = len(param_grid2)
    if verbose:
        print("Total no. of parameter combinations:",N_params)

    ## Fitting finally (NOTE: Can be optimized to not check for unneccessary values)
    max_frac = -1
    min_dev = np.inf
    for i in range(N_params):

        if i==0:
            st = time.time()   # To get a time estimate

        # reformatting initial parameters and computing trajectories
        p = param_grid2.loc[i]
        vx0 = p['vxy0']*np.cos(p['vxy_ang0'])
        vy0 = p['vxy0']*np.sin(p['vxy_ang0'])
        rt,vt,rt_spherical,vt_spherical = falling_trajectory(p['x0'],p['y0'],p['z0'],vx0,vy0,p['vz0'],Ms_val
                                                             ,ang_range = 1*np.pi,ang_step = np.pi/100,verbose=False)
        x_t_full = au2arcsec(rt[0],dist)  # R.A. in arcsec
        y_t_full = au2arcsec(rt[1],dist)  # Decl. in arcsec
        vz_t_full = (vt[2]+svel)*1e3   # R.V. in m/s
        pctraj_full = np.array([x_t_full,y_t_full,vz_t_full]) # trajectory for full orbit

        # Selecting portion of trajectory within the area of streamer
        # Note: I did not put a condition on vel. because it is not used in dist. metric and dynamics may change
        iaxis = 0
        con1 = (pctraj_full[iaxis]>=min(pccoords[iaxis])) & (pctraj_full[iaxis]<=max(pccoords[iaxis]))
        iaxis = 1
        con2 = (pctraj_full[iaxis]>=min(pccoords[iaxis])) & (pctraj_full[iaxis]<=max(pccoords[iaxis]))
        pctraj=pctraj_full[:,con1&con2]


        if pctraj.shape[1]>2:   #check if some traj. even left within the streamer field       
            # Interpolating trajectory to same "distances" (used as independent param.) as means
            # Note: Sometimes the function extrapolates too (for smaller distances), not always a good one
            r_t,theta_t = spherical_coords(pctraj[0],pctraj[1])
            theta_ref_t = np.median(theta_t[r_t<np.percentile(r_t,r_percentile_thresh)])
            theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref_t))  # Calculating new theta_ref for traj, can avoid weirdnesses
        #     theta2_t = np.pi-np.abs(np.pi-np.abs(theta_t-theta_ref))  # Using same theta_ref as observations, right thing to do?
            d_t = r_t*np.sqrt(1+(theta_weight*theta2_t)**2)
            p = np.argsort(d_t)
        #         pdeg = N_elements-1#6#(N_elements//2)
        #         coeffs = np.polyfit(d_t[p],pctraj[0][p], deg=pdeg)  # Fit quadratic function to R.A./Decl.
        #         spx = np.poly1d(coeffs)
        #         coeffs = np.polyfit(d_t[p],pctraj[1][p], deg=pdeg)  # Fit quadratic function to R.A./Decl.
        #         spy = np.poly1d(coeffs)
        #         coeffs = np.polyfit(d_t[p],pctraj[2][p], deg=pdeg)  # Fit quadratic function to R.A./Decl.
        #         spz = np.poly1d(coeffs)
        #     spx = CubicSpline(d_t[p], pctraj[0][p],bc_type='natural')#, s=s)
        #     spy = CubicSpline(d_t[p], pctraj[1][p],bc_type='natural')#, s=s)
        #     spz = CubicSpline(d_t[p], pctraj[2][p],bc_type='natural')#, s=s)
        #     kspline = 5
        #     spx = UnivariateSpline(d_t[p], pctraj[0][p],k=kspline)#, s=s)
        #     spy = UnivariateSpline(d_t[p], pctraj[1][p],k=kspline)#, s=s)
        #     spz = UnivariateSpline(d_t[p], pctraj[2][p],k=kspline)#, s=s)
            spx = interp1d(d_t[p], pctraj[0][p],fill_value='extrapolate',kind='slinear')#, s=s)
            spy = interp1d(d_t[p], pctraj[1][p],fill_value='extrapolate',kind='slinear')#, s=s)
            spz = interp1d(d_t[p], pctraj[2][p],fill_value='extrapolate',kind='slinear')#, s=s)
            traj_comp = np.array([spx(d_means),spy(d_means),spz(d_means)])
        else:
            traj_comp = np.full(pcmeans.shape,np.nan)  # Just nan array of same shape

        res = pcmeans-traj_comp
        res_norm_sq = (res/pcstds)**2
        fit_frac = np.sum((res_norm_sq)<1)/3/N_elements
        chi2 = np.sum(res_norm_sq)
        param_grid2.loc[i,'deviation'] = chi2  # In principle, deviation doesn't have to be chi2, maybe a median dev. be better?
        param_grid2.loc[i,'fit_fraction'] = fit_frac
        if param_grid2.loc[i,'fit_fraction']>max_frac:
            i_m=i
            max_frac=param_grid2.loc[i,'fit_fraction']
            min_dev=param_grid2.loc[i,'deviation']
            traj_m = [rt,vt,rt_spherical,vt_spherical]
            traj_comp_m = traj_comp
        elif param_grid2.loc[i,'fit_fraction']==max_frac:
            if param_grid2.loc[i,'deviation']<min_dev:
                i_m=i
                min_dev=param_grid2.loc[i,'deviation']
                traj_m = [rt,vt,rt_spherical,vt_spherical]
                traj_comp_m = traj_comp

        if i==0:
            exec_time = time.time() - st   # To get a time estimate
            print("Expected time for fitting [mins]:",np.round(exec_time*N_params/60,1))
    param_m = param_grid2.iloc[i_m]
      
    if verbose:
        print("Time taken for fitting [mins]:",np.round((time.time()-st)/60,1))
        print("The best solution (combination index {}):\n Fraction of fitted values: {:.2f}\n Deviation: {:.2f}\n z0 [au]: {:.2f}\n vxy0 [km/s]: {:.2f} and vxy_ang0 [degree]: {:.4f} (vx0: {:.2f} and vy0: {:.2f})"
              .format(i_m,param_m['fit_fraction'],param_m['deviation'],param_m['z0'],param_m['vxy0'],param_m['vxy_ang0']*180/np.pi
                      ,param_m['vxy0']*np.cos(param_m['vxy_ang0']),param_m['vxy0']*np.sin(param_m['vxy_ang0'])))

                
    if show_fit_cost:
        # Note: maybe a imshow like triangular figure (with 3 axis) will be nice
        ## Distribution of fitting fraction wrt vxy_ang0
        ang_means = param_grid2.groupby('vxy_ang0').mean()
        ang_stds = param_grid2.groupby('vxy_ang0').std()
        plt.errorbar(ang_means.index*180/np.pi,ang_means.fit_fraction,yerr=ang_stds.fit_fraction,fmt='o')
        plt.plot(param_m['vxy_ang0']*180/np.pi,param_m.fit_fraction,'rx',markersize=15)
        plt.xlabel('Orientation of initial P.O.S. velocity [degree]')
        plt.ylabel('Mean fitting fraction')
        plt.show()

        grid2display = param_grid2[param_grid2['vxy_ang0']==param_m['vxy_ang0']]
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')  # Nan values will show as red
        
        ## Distribution of fitting fraction wrt z0 and vxy0
        cfield = 'fit_fraction'
        pivot_df = grid2display.pivot(index='z0', columns='vxy0', values=cfield)
        pivot_df.sort_index(axis=0,inplace=True,ascending=True)
        pivot_df.sort_index(axis=1,inplace=True,ascending=True)
        plt.imshow(pivot_df,extent = [pivot_df.columns[0]-vxy0_step/2,pivot_df.columns[-1]+vxy0_step/2
                                      ,pivot_df.index[-1]+z0_step/2,pivot_df.index[0]-z0_step/2],aspect='auto',cmap=cmap)
        plt.plot(param_m['vxy0'],param_m['z0'],'rx',markersize = 20)
        plt.ylabel('Initial radial distance [au]')
        plt.xlabel('Initial POS speed [km/s]')
        plt.colorbar(label='Fitting fraction')
        plt.show()

        ## Distribution of deviation (loglog scaled) wrt z0 and vxy0
        cfield = 'loglogdev'
        grid2display['loglogdev'] = np.log10(np.log10(grid2display['deviation']))
        pivot_df = grid2display.pivot(index='z0', columns='vxy0', values=cfield)
        pivot_df.sort_index(axis=0,inplace=True,ascending=True)
        pivot_df.sort_index(axis=1,inplace=True,ascending=True)
        plt.imshow(pivot_df,extent = [pivot_df.columns[0]-vxy0_step/2,pivot_df.columns[-1]+vxy0_step/2
                                      ,pivot_df.index[-1]+z0_step/2,pivot_df.index[0]-z0_step/2],aspect='auto',cmap=cmap)
        plt.plot(param_m['vxy0'],param_m['z0'],'rx',markersize = 20)
        plt.ylabel('Initial radial distance [au]')
        plt.xlabel('Initial POS speed [km/s]')
        plt.colorbar(label='log(log(deviation))')
        plt.show()
        
    if show_fit_3d:
        x_t = au2arcsec(traj_m[0][0],dist)
        y_t = au2arcsec(traj_m[0][1],dist)
        vz_t = (traj_m[1][2]+svel)*1e3
        pctraj_m = np.array([x_t,y_t,vz_t]) # trajectory in same format as everything else
        fit_3d_plot(pccoords,pcmeans,pcstds,flux,pctraj_m,traj_comp_m,fit2html,html_path)          
    
    return(param_grid2,traj_m,[traj_comp_m,pcmeans,pcstds])

def fit_3d_plot(pccoords,pcmeans=None,pcstds=None,flux=None,traj=None,traj_interp=None,fit2html=False,html_path=None):

    '''
    Code to visualize pointcloud, observational curve, theoretical curve for streamers in PPV space.
    
    Args:
    pccoords: 3xN array containing coordinates of all the points in streamer
    pcmeans: 3xN_elements array containing coordinates of the means of points in streamer
    pcstds: 3xN_elements array containing coordinates of the standard deviations of points in streamer
    flux: N_elements long array containing flux values for observed streamer (used for colour)
    traj: 3xN1 array containing coordinates for a theoretical trajectory
    traj_interp: 3xN_elements array containing interpolated (or some extrapolated) coordinates for a theoretical trajectory
    fit2html: Whether to save 3D fitting plot to .html file
    html_path: Path to save 3D fitting plot to .html file        
    '''
    
    data2plt = []
    if traj is not None:
        data2plt+=[go.Scatter3d(x=traj[0], y=traj[1], z=traj[2],mode='lines',name='Trajectory',
                       marker=dict(size=7,color='black',opacity=0.5)
                       )]

    if traj_interp is not None:
        data2plt+=[go.Scatter3d(x=traj_interp[0], y=traj_interp[1], z=traj_interp[2],mode='markers',name='Trajectory (interpolated)',
                       marker=dict(size=7,color='black',opacity=1)
                       )]
        
    
    if pcmeans is not None:
        if pcstds is None:
            pcstds = np.zeros(pcmeans.shape)
        data2plt+=[go.Scatter3d(x=pcmeans[0], y=pcmeans[1], z=pcmeans[2],mode='lines+markers',name='Means',
                       error_x=go.scatter3d.ErrorX(array=pcstds[0]),
                       error_y=go.scatter3d.ErrorY(array=pcstds[1]),
                       error_z=go.scatter3d.ErrorZ(array=pcstds[2]),
                       marker=dict(size=9,color='red',opacity=1,symbol='square')
                   )]
    
    N_points = len(pccoords[0])
    opacity = min(0.5,1500/N_points)  # Opacity should be higher if less data points
    if flux is None:
        color = 'blue'
        colorbar = dict()
    else:
        color = flux
        colorbar = dict(title="Flux")
    data2plt+=[go.Scatter3d(x=pccoords[0], y=pccoords[1], z=pccoords[2],mode='markers',name='Observations',
           marker=dict(size=2,color=color,opacity=opacity,colorbar=colorbar)
           )]

    layout = go.Layout(scene=dict(xaxis_title='R.A. offset [arcsec]',
                          yaxis_title='Decl. offset [arcsec]',
                          zaxis_title='Radial vel. offset [m/s]',
                          xaxis_range=[np.nanmin(pccoords[0]), np.nanmax(pccoords[0])],
                          yaxis_range=[np.nanmin(pccoords[1]), np.nanmax(pccoords[1])],
                          zaxis_range=[np.nanmin(pccoords[2]), np.nanmax(pccoords[2])],
                          aspectmode='cube'),
    #                       margin=dict(l=0, r=0, b=0, t=0), 
               margin=dict(l=0, r=0, b=0, t=0), showlegend=True,
               )

    fig = go.Figure(data=data2plt, layout=layout)
    fig.update_layout(legend_orientation="h")

    if fit2html:
        if html_path == None:
            html_path = str(input("Path to store html 3D plot:"))
        fig.write_html(html_path, include_plotlyjs='cdn')
    fig.show()
    
def streamer_subcube(original_cube,xmin,xmax,ymin,ymax,vmin,vmax,rms_thresh=3):

    '''
    Code to get a subset of the original cube (subcube) for the given limits
    
    Args:
    original_cube: SpectralCube object from which to get the subcube
    xmin: Min. R.A. offset in arcsec
    xmax: Max. R.A. offset in arcsec
    ymin: Min. Decl. offset in arcsec
    ymax: Max. Decl. offset in arcsec
    vmin: Min. radial vel. in same units as spectral axis of cube
    vmax: Max. radial vel. in same units as spectral axis of cube
    rms_thresh: This will be multiplied to the median-absolute-deviation of the subcube fluxes to remove low flux pixels
    
    Returns:
    scube_cleaned: SpectralCube object with only the dominant cluster of emisssion (usually cleaned streamer emission)
    '''
    
    info_header = original_cube.header  # header of the cube, contains required information
    x_conv_fac = 1/60/60/info_header['CDELT1']
    xmin_p = int((xmin*x_conv_fac)+info_header['CRPIX1'])
    xmax_p = int((xmax*x_conv_fac)+info_header['CRPIX1'])
    y_conv_fac = 1/60/60/info_header['CDELT2']
    ymin_p = int((ymin*y_conv_fac)+info_header['CRPIX2'])
    ymax_p = int((ymax*y_conv_fac)+info_header['CRPIX2'])
    vunit = original_cube.spectral_axis.unit
    ## Note: a check can be added to see if requested limits are within the limits of the cube itself
    
    original_cubev = original_cube.spectral_slab(vmin*vunit,vmax*vunit)    # Selecting velocities
    original_cubevc = original_cubev[:,min(ymin_p,ymax_p):max(ymin_p,ymax_p)   
                     ,min(xmin_p,xmax_p):max(xmin_p,xmax_p)]   # Selecting pixels
#     print(min(ymin_p,ymax_p),max(ymin_p,ymax_p),min(xmin_p,xmax_p),max(xmin_p,xmax_p)) 
    subcube = original_cubevc.with_mask(original_cubevc > rms_thresh*original_cubevc.mad_std())  # Removing low flux values 
    return(subcube)

def streamer_cleaning(scube,use_scaled_inds=False,model=None,show_cluster=True):

    '''
    Code to remove random blobs of emission in the streamer subcube, using clustering algorithm
    
    Args:
    scube: SpectralCube object (subcube) with streamer emission
    use_scaled_inds (Optional[bool]): Whether to use scaled indices for cluster identification
    model (Optional): sklearn.cluster type model to be used for cluster identification
    show_cluster (Optional[bool]): Whether to show 3D plot with main cluster marked
    
    Returns:
    scube_cleaned: SpectralCube object with only the dominant cluster of emisssion (usually cleaned streamer emission)
    
    '''
    
    pcloud = np.array(scube)
    rms_mask = ~np.isnan(pcloud)
    pcinds = np.indices(pcloud.shape)#.shape   #Array of indices of all points in streamer point cloud
    pcz = pcinds[0][rms_mask]    #Individual indices with high enough SNR
    pcy = pcinds[1][rms_mask]
    pcx = pcinds[2][rms_mask]
    pc_inds=np.array([pcx,pcy,pcz]).T
    
    if use_scaled_inds:   # if you want to scale indices first, usually not good result
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        pc_inds_s = scaler.fit_transform(pc_inds)
        pc_inds2 = pc_inds_s
    else:
        pc_inds2 = pc_inds
    
    if model is None:  # Default clustering model is OPTICS
        from sklearn.cluster import OPTICS
        model = OPTICS()
        
    # fit model and predict clusters
    yhat = model.fit_predict(pc_inds2)
    # retrieve unique clusters
    clusters = np.unique(yhat, return_counts=True)
    cinds = (yhat==clusters[0][np.argmax(clusters[1])])
    # show clustering result
    if show_cluster:
        pc=[go.Scatter3d(x=pc_inds[:,0][cinds], y=pc_inds[:,1][cinds], z=pc_inds[:,2][cinds],name="Streamer",mode='markers',
               marker=dict(size=2,color='red',opacity=0.6)
               )]
        pc2=[go.Scatter3d(x=pc_inds[:,0][~cinds], y=pc_inds[:,1][~cinds], z=pc_inds[:,2][~cinds],name="Random blobs?",mode='markers',
               marker=dict(size=2,color='black',opacity=0.6)
               )]
        fig = go.Figure(data=pc+pc2)
        fig.show()
    
    # applying cluster masking on streamer subcude
    inds_cluster = pc_inds[cinds]
    cmask = np.full(pcloud.shape,False)
    for ind in inds_cluster:
        cmask[ind[2],ind[1],ind[0]] = True
    scube_cleaned = scube.with_mask(cmask)
    # scube_cleaned.write(fits_fil[:-5]+'_streamer.fits',format = 'fits',overwrite = True)
    return(scube_cleaned)
    
def parameter_errors(params, criteria='fit_fraction', threshold=0.90,min_resolution=True
                      ,seperate_vxy=True,mean_replace=True):
    '''
    Code to estimate uncertainities of free parameters used for TIPSY fitting. 
    
    Args:
    params: Pandas DataFrame with all the parameter combinations used and corresponding deviations
    criteria (Optional[str]): Quantity to be used to select good-enough fits for errors
    threshold (Optional[float]): Lower limiting value of 'criteria', to select good-enough fits for errors
    min_resolution (Optional[bool]): If True, replaces calculated errors less than the fitting resolution, with the fitting resolution
    seperate_vxy (Optional[bool]): If True, also provides errors for speed in R.A. and Decl. (vx0 and vy0), using error propogation
    mean_replace (Optional[bool]): If True, replaces the 'value' calculated as means of distributions of free parameters 
    with the parameter combination for the best overall fit (maximised fit_fraction, minimized chi2).
    This is more useful if you want to use the best parameter combination for future analysis.
    
    Returns:
    vals: Pandas DataFrame with representative values and standard deviations (errors) for all the free parameters for good-enough fits
'''
    from uncertainties import ufloat,umath
    
    paramsg = params[params[criteria]>threshold]  # Good-enough fits

#     params2 = params[params.fit_fraction == params.fit_fraction.max()]
#     params3 = params2[params2.deviation == params2.deviation.min()]  # best parameter combination
#     bools = [[]]*len(pnames)    # positions where each parameter is same as the one for the best overall solution
#     bools[0] = (paramsg[pnames[0]] == params3[pnames[0]].median())
#     bools[1] = (paramsg[pnames[1]] == params3[pnames[1]].median())
#     bools[2] = (paramsg[pnames[2]] == params3[pnames[2]].median())
#     arrs = [[]]*len(pnames)   # Fix two parameters and get values for the third parameter
#     arrs[0] = paramsg[bools[2] & bools[1]][pnames[0]]
#     arrs[1] = paramsg[bools[2] & bools[0]][pnames[1]]
#     arrs[2] = paramsg[bools[0] & bools[1]][pnames[2]]

    pnames = ['vxy0','vxy_ang0','z0']  # (free) parameters to consider, could be a free parameter in future
    vals = pd.DataFrame(index=pnames,columns=['value','error'])

    for i in range(len(pnames)):
        q=paramsg[pnames[i]]
        vals.loc[pnames[i],'value'] = np.nanmean(q)
        vals.loc[pnames[i],'error'] = np.nanstd(q)
        
    if mean_replace: # replace calculated means with the parameter combination for the best fit
        params2 = params[params.fit_fraction == params.fit_fraction.max()]
        params3 = params2[params2.deviation == params2.deviation.min()]  # best parameter combination
        vals.loc[pnames,'value'] = params3[pnames].median()

    if min_resolution:
        # if calculated errors are less that resolution used in fitting, replace them with the resolution
        min_errors = [np.nan]*len(pnames)
        for i in range(len(pnames)):
            temp = np.sort(params[pnames[i]].unique()) # sorted array of unique values
            min_errors[i] = np.abs(temp[1]-temp[0])  # Diff. between second-smallest and smallest values
        vals.error = np.where(vals.error<min_errors, min_errors, vals.error)
        
    if seperate_vxy:
        u_vxy_ang0 = ufloat(vals.loc['vxy_ang0','value'],vals.loc['vxy_ang0','error']) # reformatting errors in uncertainties format
        u_vxy0 = ufloat(vals.loc['vxy0','value'],vals.loc['vxy0','error'])
        u_vx0 = u_vxy0*umath.cos(u_vxy_ang0)  # Using uncertainties package for easy error propogations, can be replaced
        u_vy0 = u_vxy0*umath.sin(u_vxy_ang0)
        vals.loc['vx0']=[u_vx0.n,u_vx0.s]
        vals.loc['vy0']=[u_vy0.n,u_vy0.s]

    return(vals)

