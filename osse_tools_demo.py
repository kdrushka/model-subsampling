# Native packages
from math import radians, degrees, sin, cos, asin, acos, sqrt
import datetime
import sys
import os
import requests

# Third-party packages for data manipulation
import numpy as np
import pandas as pd
import xarray as xr

# Third-party packages for data interpolation
from scipy import interpolate
from xgcm import Grid

# Third-party packages for data visualizations
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


# import s3fs

from netrc import netrc
from urllib import request
from platform import system
from getpass import getpass
from http.cookiejar import CookieJar
from os.path import expanduser, join
import datetime
import gsw as sw
import numpy as np
import xgcm.grid
import netCDF4 as nc4

# ***This library includes*** 
# - setup_earthdata_login_auth
# - download_llc4320_data
# - compute_derived_fields
# - get_survey_track
# - survey_interp
# - great_circle

    
def get_survey_track_demo(ds, SAMPLING_STRATEGY, sampling_details):
     
    """
    Returns the track (lat, lon, depth, time) and indices (i, j, k, time) of the 
    sampling trajectory based on the type of sampling (SAMPLING_STRATEGY), 
    and sampling details (in dict sampling_details), which includes
    number of days, waypoints, and depth range, horizontal and vertical platform speed
    -- these can be typical values (default) or user-specified (optional)
    """
    
    # Change time from datetime to integer
    ds = ds.assign_coords(time=np.linspace(0,ds.time.size-1, num=ds.time.size)) # time is now in hours

    # Convert lon, lat and z to index i, j and k with f_x, f_y and f_z
    # XC, YC and Z are the same at all times, so select a single time
    X = ds.XC.isel(time=0) 
    Y = ds.YC.isel(time=0)
    i = ds.i
    j = ds.j
    z = ds.Z.isel(time=0)
    k = ds.k
    f_x = interpolate.interp1d(X[0,:].values, i)
    f_y = interpolate.interp1d(Y[:,0].values, j)
    f_z = interpolate.interp1d(z, k, bounds_error=False)

    # Get boundaries and center of model region
    model_boundary_n = Y.max().values
    model_boundary_s = Y.min().values
    model_boundary_w = X.min().values
    model_boundary_e = X.max().values
    model_xav = ds.XC.isel(time=0, j=0).mean(dim='i').values
    model_yav = ds.YC.isel(time=0, i=0).mean(dim='j').values
    # --------- define sampling -------
    survey_time_total = (ds.time.values.max() - ds.time.values.min()) * 3600 # (seconds) - limits the survey to a total time
    
    # defaults:
    AT_END = 'terminate' # behaviour at and of trajectory: 'repeat' or 'terminate'. (could also 'restart'?)
    
    
    # typical speeds and depth ranges based on platform 
    if SAMPLING_STRATEGY == 'sim_uctd':
        PATTERN = sampling_details['PATTERN']
        # typical values for uctd sampling:
        zrange = [-5, -500] # depth range of profiles (down is negative)
        hspeed = 5 # platform horizontal speed in m/s
        vspeed = 1 # platform vertical (profile) speed in m/s (NOTE: may want different up/down speeds)  
    elif SAMPLING_STRATEGY == 'sim_glider':
        PATTERN = sampling_details['PATTERN']
        zrange = [-1, -1000] # depth range of profiles (down is negative)
        hspeed = 0.25 # platform horizontal speed in m/s
        vspeed = 0.1 # platform vertical (profile) speed in m/s  (NOTE: is this typical?)  
    elif SAMPLING_STRATEGY == 'trajectory_file':
        # load file
        traj = xr.open_dataset(sampling_details['trajectory_file'])
        xwaypoints = traj.xwaypoints.values
        ywaypoints = traj.ywaypoints.values
        zrange = traj.zrange.values # depth range of profiles (down is negative)
        hspeed = traj.hspeed.values # platform horizontal speed in m/s
        vspeed = traj.vspeed.values # platform vertical (profile) speed in m/s
        PATTERN = traj.attrs['pattern']
        
    
    # specified sampling always overrides the defaults: 
    if sampling_details['zrange'] is not None:
        zrange = sampling_details['zrange']     
    if sampling_details['hspeed'] is not None:
        hspeed = sampling_details['hspeed']   
    if sampling_details['vspeed'] is not None:
        vspeed = sampling_details['vspeed'] 
    if sampling_details['AT_END'] is not None:
        AT_END = sampling_details['AT_END'] 
        
    # define x & y waypoints and z range
    # xwaypoints & ywaypoints must have the same size
    if PATTERN == 'lawnmower':
        # "mow the lawn" pattern - define all waypoints
        if not(SAMPLING_STRATEGY == 'trajectory_file'):
            # generalize the survey for this region
            xwaypoints = model_boundary_w + 1 + [0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2]
            ywaypoints = model_boundary_s + [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
    elif PATTERN == 'back-forth':
        if not(SAMPLING_STRATEGY == 'trajectory_file'):
            # repeated back & forth transects - define the end-points
            xwaypoints = model_xav + [-1, 1]
            ywaypoints = model_yav + [-1, 1]
        # repeat waypoints based on total # of transects: 
        dkm_per_transect = great_circle(xwaypoints[0], ywaypoints[0], xwaypoints[1], ywaypoints[1]) # distance of one transect in km
        t_per_transect = dkm_per_transect * 1000 / hspeed # time per transect, seconds
        num_transects = np.round(survey_time_total / t_per_transect)
        for n in np.arange(num_transects):
            xwaypoints = np.append(xwaypoints, xwaypoints[-2])
            ywaypoints = np.append(ywaypoints, ywaypoints[-2])
        
    # time resolution of sampling (dt):
    # for now, use a constant  vertical resolution (can change this later)
    zresolution = 1 # meters
    zprofile = np.arange(zrange[0],zrange[1],-zresolution) # depths for one profile
    ztwoway = np.append(zprofile,zprofile[-1:0:-1])

    dt = zresolution / vspeed # sampling resolution in seconds
    # for each timestep dt 
    deltah = hspeed*dt # horizontal distance traveled per sample
    deltav = vspeed*dt # vertical distance traveled per sample

    # determine the sampling locations in 2-d space
    # - initialize sample locations xs, ys, zs, ts
    xs = []
    ys = []
    zs = []
    ts = []
    dkm_total = 0 
    
    
    for w in np.arange(len(xwaypoints)-1):
        # interpolate between this and the following waypoint:
        dkm = great_circle(xwaypoints[w], ywaypoints[w], xwaypoints[w+1], ywaypoints[w+1])
        # number of time steps (vertical measurements) between this and the next waypoint
        nstep = int(dkm*1000 / deltah) 
        yi = np.linspace(ywaypoints[w], ywaypoints[w+1], nstep)
        xi = np.linspace(xwaypoints[w], xwaypoints[w+1], nstep)
        xi = xi[0:-1] # remove last point, which is the next waypoint
        xs = np.append(xs, xi) # append
        yi = yi[0:-1] # remove last point, which is the next waypoint
        ys = np.append(ys, yi) # append
        dkm_total = dkm_total + dkm
        t_total = dkm_total * 1000 / hspeed # cumulative survey time to this point
        # cut off the survey after survey_time_total, if specified
        if t_total > survey_time_total:
            break
            
    # if at the end of the waypoints but time is less than the total, trigger AT_END behavior:
    if t_total < survey_time_total:
        if AT_END == 'repeat': 
            # start at the beginning again
            # determine how many times the survey repeats:
            num_transects = np.round(survey_time_total / t_total)
            xtemp = xs
            ytemp = ys
            # ***** HAVE TO ADD THE TRANSECT BACK TO THE START !!!
            for n in np.arange(num_transects):
                xs = np.append(xs, xtemp)
                ys = np.append(ys, ytemp)
        elif AT_END == 'reverse': 
            # turn around & go in the opposite direction
            # determine how many times the survey repeats:
            num_transects = np.round(survey_time_total / t_total)
            
            xtemp = xs
            ytemp = ys
            # append both a backward & another forward transect
            for n in np.arange(np.ceil(num_transects/2)):
                xs = np.append(np.append(xs, xtemp[-2:1:-1]), xtemp)
                ys = np.append(np.append(ys, ytemp[-2:1:-1]), ytemp)


    
    # depths: repeat (tile) the two-way sampling depths (NOTE: for UCTD sampling, often only use down-cast data)
    # how many profiles do we make during the survey?
    n_profiles = np.ceil(xs.size / ztwoway.size)
    zs = np.tile(ztwoway, int(n_profiles))
    zs = zs[0:xs.size]
    # sample times: (units are in seconds since zero => convert to days, to agree with ds.time)
    ts = dt * np.arange(xs.size) / 86400 
    
    # get rid of points with sample time > survey_time_total
    if survey_time_total > 0:
        idx = np.abs(ts*86400 - survey_time_total).argmin() # index of ts closest to survey_time_total
        xs = xs[:idx]
        ys = ys[:idx]
        ts = ts[:idx]
        zs = zs[:idx]
        
    ## Assemble dataset:
    # real (lat/lon) coordinates
    survey_track = xr.Dataset(
        dict(
            lon = xr.DataArray(xs,dims='points'),
            lat = xr.DataArray(ys,dims='points'),
            dep = xr.DataArray(zs,dims='points'),
            time = xr.DataArray(ts,dims='points')
        )
    )
    # transform to i,j,k coordinates:
    survey_indices= xr.Dataset(
        dict(
            i = xr.DataArray(f_x(survey_track.lon), dims='points'),
            j = xr.DataArray(f_y(survey_track.lat), dims='points'),
            k = xr.DataArray(f_z(survey_track.dep), dims='points'),
            time = xr.DataArray(survey_track.time, dims='points'),
        )
    )
    
    # return details about the sampling (mostly for troubleshooting)
    # could prob do this with a loop
    sampling_parameters = {
        'PATTERN' : PATTERN, 
        'zrange' : zrange,
        'hspeed' : hspeed,
        'vspeed' : vspeed,
        'dt_sample' : dt
    
}

    
    return survey_track, survey_indices, sampling_parameters

def survey_interp_demo(ds, survey_track, survey_indices):
    """
    interpolate dataset 'ds' along the survey track given by 
    'survey_indices' (i,j,k coordinates used for the interpolation), and
    'survey_track' (lat,lon,dep,time of the survey)
    
    Returns:
        subsampled_data: all field interpolated onto the track
        sh_true: 'true' steric height along the track
    
    """
      
        
    ## Create a new dataset to contain the interpolated data, and interpolate
    subsampled_data = xr.Dataset() # NOTE: add more metadata to this dataset?
    subsampled_data['Theta']=ds.Theta.interp(survey_indices) # NOTE: is there a smarter way to do this using variable names and a loop?
    subsampled_data['Salt']=ds.Salt.interp(survey_indices) 
    subsampled_data['steric_height']=ds.steric_height.interp(survey_indices) 
    subsampled_data['vorticity']=ds.vorticity.interp(survey_indices) 
    subsampled_data['lon']=survey_track.lon
    subsampled_data['lat']=survey_track.lat
    subsampled_data['dep']=survey_track.dep
    subsampled_data['time']=survey_track.time    
        
    # 2-d interpolation of steric height "truth" - just along the surevy track, full depth interpolation
    survey_indices_2d =  survey_indices.drop_vars('k') # create 2-d survey track by removing the depth dimension
    sh_true = ds.steric_height.isel(k=-1).interp(survey_indices_2d) # interpolate full-depth SH (last value of dep, i.e., k=-1) to the 2-d track

    
    ## Get u, v
    grid = Grid(ds, coords={'X':{'center': 'i', 'left': 'i_g'}, 
                     'Y':{'center': 'j', 'left': 'j_g'},
                     'Z':{'center': 'k'}})

    # Interpolate U and V from i_g, j_g to i, j 
    ### Interpolate variables that are not on the i-j grid, but shifted. 
    # Roughly based on: https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
    U_c = grid.interp(ds.U, 'X', boundary='extend')
    V_c = grid.interp(ds.V, 'Y', boundary='extend')

    # Compute vorticity and interpolate to i,j
    vorticity = (grid.diff(ds.V*ds.DXG, 'X') - grid.diff(ds.U*ds.DYG, 'Y'))/ds.RAZ
    vorticity = grid.interp(grid.interp(vorticity, 'X', boundary='extend'), 'Y', boundary='extend')
    
    ## Interpolate and add to subsampled_data
    subsampled_data['U'] = U_c.interp(survey_indices)
    subsampled_data['V'] = V_c.interp(survey_indices)
    subsampled_data['vorticity'] = vorticity.interp(survey_indices)
    
    return subsampled_data, sh_true



# great circle distance (from Jake Steinberg) 
def great_circle(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
