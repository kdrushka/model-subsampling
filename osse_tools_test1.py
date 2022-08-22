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


from netrc import netrc
from urllib import request
from platform import system
from getpass import getpass
from http.cookiejar import CookieJar
from os.path import expanduser, join
from datetime import datetime, date, time, timedelta
import gsw as sw
import numpy as np
import xgcm.grid
import netCDF4 as nc4


#MB
import matplotlib.dates as mdates
import s3fs
import numba
from numba import jit
import fastai
from django.urls import path
from fastai.imports import *

# ***This library includes*** 
# - setup_earthdata_login_auth
# - download_llc4320_data
# - compute_derived_fields
# - get_survey_track
# - survey_interp
# - great_circle
"""Set up the the Earthdata login authorization for downloading data.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: True if succesful, False otherwise.
        
    Raises: 
        FileNotFoundError: If the Earthdata account details entered are incorrect.
    

    """

def setup_earthdata_login_auth(endpoint: str='urs.earthdata.nasa.gov'):
    """Set up the the Earthdata login authorization for downloading data.

    Extended description of function.

    Returns:
        bool: True if succesful, False otherwise.
        
    Raises: 
        FileNotFoundError: If the Earthdata account details entered are incorrect.
    

    """
    return True
    netrc_name = "_netrc" if system()=="Windows" else ".netrc"
    try:
        username, _, password = netrc(file=join(expanduser('~'), netrc_name)).authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        print('Please provide your Earthdata Login credentials for access.')
        print('Your info will only be passed to %s and will not be exposed in Jupyter.' % (endpoint))
        username = input('Username: ')
        password = getpass('Password: ')
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)
    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)
    
"""Set up the the Earthdat login authorization for downloading data.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: True if succesful, False otherwise.
        
    Raises: 
        FileNotFoundError: If the Earthdata account details entered are incorrect.
    

    """
def rotate_vector_to_EN(U, V, AngleCS, AngleSN):
    """Rotate vector to east north direction.
    
    Assumes that AngleCS and AngleSN are already of same dimension as V and U (i.e. already interpolated to cell center)
                
    Args:
        U (xarray Dataarray): Zonal vector component
        V (array Dataarray): Meridonal vector component

    Returns:
        uE (xarray Dataarray): TRotated zonal component
        vN (xarray Dataarray): Rotated meridonial component
        
    Raises: 
        FileNotFoundError: If the Earthdata account details entered are incorrect.
    
    Note: adapted from https://github.com/AaronDavidSchneider/cubedsphere/blob/main/cubedsphere/regrid.py
    

    """
               
                # rotate the vectors:
                uE = AngleCS * U - AngleSN * V
                vN = AngleSN * U + AngleCS * V

                return uE, vN
            

def download_llc4320_data(RegionName, datadir, start_date, ndays):
    """Download the MITgcm LLC4320 data from PODAAC Earthdata website.

    It creates a http access for each target file using the setup_earthdata_login_auth function. It checks for existing llc4320 files in 'datadir' and downloads them in the datadir if not found.

    Args:
        RegionName (str): It can be selected from WesternMed, ROAM_MIZ, NewCaledonia, NWPacific, BassStrait, RockallTrough, ACC_SMST, MarmaraSea, LabradorSea, CapeBasin
        datadir (str): Directory where input models are stored
        start_date (datetime): Starting date for downloading data
        ndays (int): Number of days to be downloaded from the start date

    Returns:
        None
        
    Raises: 
        FileNotFoundError: If the Earthdata account details entered are incorrect
        error-skipping this file: If the file already exists
    

    """
   
    ShortName = "MITgcm_LLC4320_Pre-SWOT_JPL_L4_" + RegionName + "_v1.0"
    date_list = [start_date + timedelta(days=x) for x in range(ndays)]
    target_files = [f'LLC4320_pre-SWOT_{RegionName}_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list of files to check for/download
    setup_earthdata_login_auth()
    
    # https access for each target_file
    url = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected"
    https_accesses = [f"{url}/{ShortName}/{target_file}" for target_file in target_files]
#     print(https_accesses)
    

    Path(datadir).mkdir(parents=True, exist_ok=True) # create datadir if it doesn't exist

    # list of dataset objects
    dds = []
    for https_access,target_file in zip(https_accesses,target_files):
        

        if not(os.path.isfile(datadir + target_file)):
            print('downloading ' + target_file) # print file name
            try:
                filename_dir = os.path.join(datadir, target_file)
                request.urlretrieve(https_access, filename_dir)
            except:
                print(' ---- error - skipping this file')


def get_survey_track(ds, sampling_details):
    
    """Calculates the survey indices and track based on the sampling details for the dataset for all days.


    Args:
        ds (xarray.core.dataset.Dataset): MITgcm LLC4320 data for all days
        sampling_details (dict): It includes number of days, waypoints, and depth range, horizontal and vertical platform speed. These can typical (default) or user-specified, in the                                      case where user specfies only some of the details the default values will be used for rest.

    Returns:
        survey_track (xarray.core.dataset.Dataset): Returns the track (lat, lon, depth, time) of the sampling trajectory based on the type of sampling                               
        survey_indices (xarray.core.dataset.Dataset): Returns the indices (i, j, k, time) of the sampling trajectory based on the type of sampling
        sampling_details (dict): Returns the modified sampling_details by filling in the missing parameters with defaults.
        
    Raises: 
        Sampling strategy is invalid: If a sampling strategy is not specified or different from the available strategies - sim_utcd, sim_glider, sim_mooring, wave_glider, sail_drone
    

    """

    
    survey_time_total = (ds.time.values.max() - ds.time.values.min()) # (timedelta) - limits the survey to a total time
    survey_end_time = ds.time.isel(time=0).data + survey_time_total # end time of survey
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
    SAMPLING_STRATEGY = sampling_details['SAMPLING_STRATEGY']
    # ------ default sampling parameters: in the dict named "defaults" -----
    defaults = {}
    # default values depend on the sampling type
    # typical speeds and depth ranges based on platform 
    if SAMPLING_STRATEGY == 'sim_uctd':
        # typical values for uctd sampling:
        defaults['zrange'] = [-5, -500] # depth range of profiles (down is negative)
        defaults['hspeed'] = 5 # platform horizontal speed in m/s
        defaults['vspeed'] = 1 # platform vertical (profile) speed in m/s (NOTE: may want different up/down speeds)  
        defaults['PATTERN'] = 'lawnmower'
        defaults['AT_END'] = 'terminate'  # behaviour at and of trajectory: 'repeat', 'reverse', or 'terminate'
    elif SAMPLING_STRATEGY == 'sim_glider':
        defaults['zrange'] = [-1, -1000] # depth range of profiles (down is negative)
        defaults['hspeed'] = 0.25 # platform horizontal speed in m/s
        defaults['vspeed'] = 0.1 # platform vertical (profile) speed in m/s     
        defaults['AT_END'] = 'terminate'  # behaviour at and of trajectory: 'repeat', 'reverse', or 'terminate'
        defaults['PATTERN'] = 'lawnmower'
        #MB
    elif SAMPLING_STRATEGY == 'wave_glider':
        defaults['zrange'] = [-1, -1.5] # depth range of profiles (down is negative)
        defaults['hspeed'] = 1 # platform horizontal speed in m/s
        defaults['vspeed'] = 0 # platform vertical (profile) speed in m/s     
        defaults['AT_END'] = 'terminate'  # behaviour at and of trajectory: 'repeat', 'reverse', or 'terminate'
        defaults['PATTERN'] = 'back-forth'
        #MB
    elif SAMPLING_STRATEGY == 'sail_drone':
        defaults['zrange'] = [-1, -3] # depth range of profiles (down is negative)
        defaults['hspeed'] = 2.57 # platform horizontal speed in m/s
        defaults['vspeed'] = 0 # platform vertical (profile) speed in m/s     
        defaults['AT_END'] = 'terminate'  # behaviour at and of trajectory: 'repeat', 'reverse', or 'terminate'
        defaults['PATTERN'] = 'back-forth'
        #MB
    elif SAMPLING_STRATEGY == 'sim_mooring' or SAMPLING_STRATEGY == 'mooring':
        defaults['xmooring'] = model_xav # default lat/lon is the center of the domain
        defaults['ymooring'] = model_yav
        defaults['zmooring_TS'] = [-1, -10, -50, -100] # depth of T/S instruments
        defaults['zmooring_UV'] = [-1, -10, -50, -100] # depth of U/V instruments
    elif SAMPLING_STRATEGY == 'trajectory_file':
        # load file
        traj = xr.open_dataset(sampling_details['trajectory_file'])
        defaults['xwaypoints'] = traj.xwaypoints.values
        defaults['ywaypoints'] = traj.ywaypoints.values
        defaults['zrange'] = traj.zrange.values # depth range of profiles (down is negative)
        defaults['hspeed'] = traj.hspeed.values # platform horizontal speed in m/s
        defaults['vspeed'] = traj.vspeed.values # platform vertical (profile) speed in m/s
        defaults['PATTERN'] = traj.attrs['pattern']
    else:
        # if SAMPLING_STRATEGY not specified, return an error
        print('error: SAMPLING_STRATEGY ' + SAMPLING_STRATEGY + ' invalid')
        return -1
    
    #
    defaults['SAVE_PRELIMINARY'] = False
    
    
    # merge defaults & sampling_details
    # - by putting sampling_details second, items that appear in both dicts are taken from sampling_details: 
    sampling_details = {**defaults, **sampling_details}

    # ----- define x/y/z/t points to interpolate to
    # for moorings, location is fixed so a set of waypoints is not needed.
    # however, for "sim_mooring", tile/repeat the sampling x/y/t to form 2-d arrays,
    # so the glider/uCTD interpolation framework can be used.
    # - and for "mooring", skip the step of interpolating to "points" and interpolate directly to the new x/y/t/z 
    if SAMPLING_STRATEGY == 'sim_mooring':
        # time sampling is one per model timestep
#         ts = ds.time.values / 24 # convert from hours to days
        ts = ds.time.values # in hours
        n_samples = ts.size
        n_profiles = n_samples
        # same sampling for T/S/U/V for now. NOTE: change this later!        
        zs = np.tile(sampling_details['zmooring_TS'], int(n_samples)) # sample depths * # of samples 
        xs = sampling_details['xmooring'] * np.ones(np.size(zs))  # all samples @ same x location
        ys = sampling_details['ymooring'] * np.ones(np.size(zs))  # all samples @ same y location
        ts = np.repeat(ts, len(sampling_details['zmooring_TS']))  # tile to match size of other fields. use REPEAT, not TILE to get the interpolation right.

#         # depth sampling - different for TS and UV
#         zs_TS = np.tile(zmooring_TS, int(n_samples))
#         zs_UV = np.tile(zmooring_UV, int(n_samples))
#         xs_TS = xmooring * np.ones(np.size(zs_TS))
#         xs_UV = xmooring * np.ones(np.size(zs_UV))
#         ys_TS = ymooring * np.ones(np.size(zs_TS))
#         ys_UV = ymooring * np.ones(np.size(zs_UV))
#         ts_TS = np.tile(ts, int(n_samples))
        
        
#         lon_TS = xr.DataArray(xs_TS,dims='points'),
#         lat_TS = xr.DataArray(ys_TS,dims='points'),
#         dep_TS = xr.DataArray(zs_TS,dims='points'),
#         time_TS = xr.DataArray(ts,dims='points')
               
#         lon = lon_TS
#         lat = lat_TS
#         dep = dep_TS
#         time = time_TS
    elif SAMPLING_STRATEGY == 'mooring':
        ts = ds.time.values # in hours
        # same sampling for T/S/U/V for now. NOTE: change this later!  
        zs = sampling_details['zmooring_TS'] 
        xs = sampling_details['xmooring']
        ys = sampling_details['ymooring']
    else:
        # --- if not a mooring, define waypoints  
    
        # define x & y waypoints and z range
        # xwaypoints & ywaypoints must have the same size
        if sampling_details['PATTERN'] == 'lawnmower':
            # "mow the lawn" pattern - define all waypoints
            if not(SAMPLING_STRATEGY == 'trajectory_file'):
                # generalize the survey for this region
                xwaypoints = model_boundary_w + 1 + [0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2]
                ywaypoints = model_boundary_s + [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
        elif sampling_details['PATTERN'] == 'back-forth':
            if not(SAMPLING_STRATEGY == 'trajectory_file'):
                # repeated back & forth transects - define the end-points
                xwaypoints = model_xav + [-1, 1]
                ywaypoints = model_yav + [-1, 1]
            # repeat waypoints based on total # of transects: 
            dkm_per_transect = great_circle(xwaypoints[0], ywaypoints[0], xwaypoints[1], ywaypoints[1]) # distance of one transect in km
#           # time per transect, seconds, as a np.timedelta64 value
            t_per_transect = np.timedelta64(int(dkm_per_transect * 1000 / sampling_details['hspeed']), 's')    
            num_transects = np.round(survey_time_total / t_per_transect)
            for n in np.arange(num_transects):
                xwaypoints = np.append(xwaypoints, xwaypoints[-2])
                ywaypoints = np.append(ywaypoints, ywaypoints[-2])
        if SAMPLING_STRATEGY == 'trajectory_file':
            xwaypoints = sampling_details['xwaypoints']
            ywaypoints = sampling_details['ywaypoints']
        # if the survey pattern repeats, add the first waypoint to the end of the list of waypoints:
        if sampling_details['AT_END'] == 'repeat': 
            xwaypoints = np.append(xwaypoints, xwaypoints[0])
            ywaypoints = np.append(ywaypoints, ywaypoints[0])                
        
## Different function
        # vertical resolution
        # for now, use a constant  vertical resolution (NOTE: could make this a variable)
        #if ((SAMPLING_STRATEGY != 'wave_glider') and (SAMPLING_STRATEGY != 'sail_drone')): 
        zresolution = 1 # meters
        # max depth can't be deeper than the max model depth in this region
        sampling_details['zrange'][1] = -np.min([-sampling_details['zrange'][1], ds.Depth.isel(time=1).max(...).values])        
        zprofile = np.arange(sampling_details['zrange'][0],sampling_details['zrange'][1],-zresolution) # depths for one profile
        ztwoway = np.append(zprofile,zprofile[-1::-1])
        # time resolution of sampling (dt):
        dt = zresolution / sampling_details['vspeed'] # sampling resolution in seconds
        dt_td64 = np.timedelta64(int(dt), 's') # np.timedelta64 format
        # for each timestep dt 
        deltah = sampling_details['hspeed']*dt # horizontal distance traveled per sample
        deltav = sampling_details['vspeed']*dt # vertical distance traveled per sample

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
            # cumulative survey time to this point, in seconds, as a np.timedelta64 value
            t_total = np.timedelta64(int(dkm_total * 1000 / sampling_details['hspeed']), 's')
            
            # cut off the survey after survey_time_total
            if t_total > survey_time_total:
                break 
                
        # km for one lap of the survey
        dkm_once = dkm_total 

        # if time is less than survey_time_total, trigger AT_END behavior:
        if t_total < survey_time_total:
            if sampling_details['AT_END'] == 'repeat': 
                # start at the beginning again
                # - determine how many times the survey repeats:
                num_transects = np.round(survey_time_total / t_total)
                x_once = xs
                y_once = ys
                for n in np.arange(num_transects):
                    xs = np.append(xs, x_once)
                    ys = np.append(ys, y_once)
                    dkm_total += dkm_once
            elif sampling_details['AT_END'] == 'reverse': 
                # turn around & go in the opposite direction
                # - determine how many times the survey repeats:
                num_transects = np.round(survey_time_total / t_total)
                x_once = xs
                y_once = ys
                # append both a backward & another forward transect
                for n in np.arange(np.ceil(num_transects/2)):
                    xs = np.append(np.append(xs, x_once[-2:1:-1]), x_once)
                    ys = np.append(np.append(ys, y_once[-2:1:-1]), y_once)
                    dkm_total += dkm_once*2


        # repeat (tile) the two-way sampling depths 
        # - number of profiles we make during the survey:
        n_profiles = np.ceil(xs.size / ztwoway.size)
        zs = np.tile(ztwoway, int(n_profiles))
        zs = zs[0:xs.size] # limit to # of sample times
        ts = ds.time.isel(time=0).data + dt_td64 * np.arange(xs.size)
        # get rid of points with sample time > survey_time_total
        if survey_time_total > 0:
            idx = np.argmin(np.abs(ts - survey_end_time))# index of ts closest to survey_end_time
            print('originally, ', idx, ' points')
            # make sure this is multiple of the # of profiles:
            idx = int(np.floor((idx+1)/len(ztwoway)) * (len(ztwoway)))
            xs = xs[:idx]
            ys = ys[:idx]
            ts = ts[:idx]
            zs = zs[:idx]
            n_profiles = np.ceil(xs.size / ztwoway.size)
            # update t_total
            t_total = np.diff(ts[[0,-1]])
            t_total_seconds = int(t_total)/1e9 # convert from nanoseconds to seconds
            # use the speed to determine dkm_total (time * hspeed)
            dkm_total = t_total_seconds * sampling_details['hspeed'] / 1000
            print('limited to ', idx, 'points: n_profiles=', n_profiles, ', ', len(zprofile), 'depths per profile, ', len(ztwoway), 'depths per two-way')
            
        sampling_details['distance_total_km'] = dkm_total
        sampling_details['time_total_s'] = t_total_seconds  
        # -- end if not a mooring
        
    # ----- Assemble dataset: -----
    # (same regardless of sampling strategy - EXCEPT "mooring")
    if not SAMPLING_STRATEGY == 'mooring':
        # - real (lat/lon) coordinates:
        survey_track = xr.Dataset(
            dict(
                lon = xr.DataArray(xs,dims='points'),
                lat = xr.DataArray(ys,dims='points'),
                dep = xr.DataArray(zs,dims='points'),
                time = xr.DataArray(ts,dims='points'),
                n_profiles = n_profiles
            )
        )
        # - transform to i,j,k coordinates:
        survey_indices= xr.Dataset(
            dict(
                i = xr.DataArray(f_x(survey_track.lon), dims='points'),
                j = xr.DataArray(f_y(survey_track.lat), dims='points'),
                k = xr.DataArray(f_z(survey_track.dep), dims='points'),
                time = xr.DataArray(survey_track.time, dims='points'),
            )
        )
    elif SAMPLING_STRATEGY == 'mooring':
        survey_track = xr.Dataset(
            dict(
                lon = xr.DataArray(xs*[1], dims='position'),
                lat = xr.DataArray(ys*[1], dims='position'),
                dep = xr.DataArray(zs, dims='depth'),
                time = xr.DataArray(ts, dims='time')

            )
        )
        # - transform to i,j,k coordinates:
        survey_indices= xr.Dataset(
            dict(
                i = xr.DataArray(f_x(survey_track.lon), dims='position'),
                j = xr.DataArray(f_y(survey_track.lat), dims='position'),
                k = xr.DataArray(f_z(survey_track.dep), dims='depth'),
                time = xr.DataArray(survey_track.time, dims='time'),
            )
        )
    # store SAMPLING_STRATEGY and DERIVED_VARIABLES in survey_track so they can be used later
    survey_track['SAMPLING_STRATEGY'] = SAMPLING_STRATEGY
#     survey_track['DERIVED_VARIABLES'] = sampling_details['DERIVED_VARIABLES']
#    survey_track['SAVE_PRELIMINARY'] = sampling_details['SAVE_PRELIMINARY']
    return survey_track, survey_indices, sampling_details
 
    
def survey_interp(ds, survey_track, survey_indices, sampling_details):
    """Interpolates dataset 'ds' along the survey track given by the sruvey coordinates.


    Args:
        ds (xarray.core.dataset.Dataset): MITgcm LLC4320 data for all days
        survey_track (xarray.core.dataset.Dataset): lat,lon,dep,time of the survey used for the interpolation
        survey_indices (xarray.core.dataset.Dataset): i,j,k coordinates used for the interpolation
        sampling_details (dict):Includes number of days, waypoints, and depth range, horizontal and vertical platform speed. These can typical (default) or user-specified, in the                                      case where user specfies only some of the details the default values will be used for rest.
        

    Returns:
        subsampled_data: all field interpolated onto the track
        sh_true: 'true' steric height along the track
        
    Raises: 
        Sampling strategy is invalid: If a sampling strategy is not specified or different from the available strategies - sim_utcd, sim_glider, sim_mooring, wave_glider, sail_drone
    

    """
      
        
    ## Create a new dataset to contain the interpolated data, and interpolate
    # for 'mooring', skip this step entirely - return an empty array for 'subsampled_data'
    SAMPLING_STRATEGY = survey_track['SAMPLING_STRATEGY']
    if SAMPLING_STRATEGY == 'mooring':
        subsampled_data = []
        
        # zgridded and times are simply zs, ta (i.e., don't interpolate to a finer grid than the mooring sampling gives)
        zgridded = survey_track['dep']
        times = survey_track['time']
        
        # -- initialize the dataset:
        sgridded = xr.Dataset(
            coords = dict(depth=(["depth"],zgridded),
                      time=(["time"],times))
        )
        # variable names (if DERIVED_VARIABLES is not set, don't load the vector quantities)
        if sampling_details['DERIVED_VARIABLES']:
            vbls3d = ['Theta','Salt','vorticity','steric_height', 'U', 'V']
            vbls2d = ['steric_height_true', 'Eta', 'KPPhbl', 'PhiBot', 'oceTAUX', 'oceTAUY', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux']
        else:
            vbls3d = ['Theta','Salt']
            vbls2d = ['Eta', 'KPPhbl', 'PhiBot', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux']
        
        
        # loop through 3d variables & interpolate:
        for vbl in vbls3d:
            print(vbl)
            sgridded[vbl]=ds[vbl].interp(survey_indices).compute().transpose()

        # loop through 2d variables & interpolate:
        # create 2-d survey track by removing the depth dimension
        survey_indices_2d =  survey_indices.drop_vars('k')
           
        
        
        for vbl in vbls2d:
            print(vbl)
            sgridded[vbl]=ds[vbl].interp(survey_indices_2d).compute()
            
        
        # clean up sgridded: get rid of the dims we don't need and rename coords
        
        if sampling_details['DERIVED_VARIABLES']:
            sgridded = sgridded.reset_coords(names = {'i', 'j', 'k'}).squeeze().rename_vars({'xav' : 'lon','yav' : 'lat'}).drop_vars(names={'i', 'j', 'k'})
        
            # for sampled steric height, we want the value integrated from the deepest sampling depth:
            sgridded['steric_height'] = (("time"), sgridded['steric_height'].isel(depth=int(len(zgridded))-1))
            # rename to "sampled" for clarity
            sgridded.rename_vars({'steric_height':'steric_height_sampled'})
        else:
            sgridded = sgridded.reset_coords(names = {'i', 'j', 'k'}).squeeze().drop_vars(names={'i', 'j', 'k'})
    
    else:
        subsampled_data = xr.Dataset(
            dict(
                t = xr.DataArray(survey_track.time, dims='points'), # call this time, for now, so that the interpolation works
                lon = xr.DataArray(survey_track.lon, dims='points'),
                lat = xr.DataArray(survey_track.lat, dims='points'),
                dep = xr.DataArray(survey_track.dep, dims='points'),
                points = xr.DataArray(survey_track.points, dims='points')
            )
        )

        # variable names (if DERIVED_VARIABLES is not set, don't load the vector quantities)
        if sampling_details['DERIVED_VARIABLES']:
            vbls3d = ['Theta','Salt','vorticity','steric_height', 'U', 'V']
            vbls2d = ['steric_height_true', 'Eta', 'KPPhbl', 'PhiBot', 'oceTAUX', 'oceTAUY', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux']
        else:
            vbls3d = ['Theta','Salt']
            vbls2d = ['Eta', 'KPPhbl', 'PhiBot', 'oceFWflx', 'oceQnet', 'oceQsw', 'oceSflux']
        
        
        print('Interpolating model fields to the sampling track...')
        # loop & interpolate through 3d variables:
        for vbl in vbls3d:
            subsampled_data[vbl]=ds[vbl].interp(survey_indices)

       

        # loop & interpolate through 2d variables:
        # create 2-d survey track by removing the depth dimension
        survey_indices_2d =  survey_indices.drop_vars('k')
        for vbl in vbls2d:
            subsampled_data[vbl]=ds[vbl].interp(survey_indices_2d)   

        # fix time, which is currently a coordinate (time) & a variable (t)
        subsampled_data = subsampled_data.reset_coords('time', drop=True).rename_vars({'t':'time'})

        # make xav and yav variables instead of coords, and rename
        if sampling_details['DERIVED_VARIABLES']:
            subsampled_data = subsampled_data.reset_coords(names = {'xav','yav'}).rename_vars({'xav' : 'lon_average','yav' : 'lat_average'})


            
        
        if sampling_details['SAVE_PRELIMINARY']:
            # ----- save preliminary data
            # (not applicable to mooring data)
            # add metadata to attributes
            attrs = sampling_details
            attrs['start_date'] = sampling_details['start_date'].strftime('%Y-%m-%d')
            end_date = subsampled_data['time'].data[-1]
            attrs['end_date'] = np.datetime_as_string(end_date,unit='D')
            attrs.pop('DERIVED_VARIABLES')    
            attrs.pop('SAVE_PRELIMINARY')
            
            # filename:
            filename_out = sampling_details['filename_out_base'] + '_subsampled.nc'
            print(f'saving to {filename_out}')
            subsampled_data.attrs = attrs
            netcdf_fill_value = nc4.default_fillvals['f4']
            dv_encoding={'zlib':True,  # turns compression on\
                        'complevel':9,     # 1 = fastest, lowest compression; 9=slowest, highest compression \
                        'shuffle':True,    # shuffle filter can significantly improve compression ratios, and is on by default \
                        'dtype':'float32',\
                        '_FillValue':netcdf_fill_value}
            # save to a new file
            subsampled_data.to_netcdf(filename_out,format='netcdf4')
            
            
            
        # -----------------------------------------------------------------------------------
        # ------Regrid the data to depth/time (3-d fields) or subsample to time (2-d fields)
        print('Gridding the interpolated data...')
        
        # if SAVE PRELIMINARY, load the saved 'subsampled_data' file without dask
        # (otherwise, subsampled_data is already in memory)
        if sampling_details['SAVE_PRELIMINARY']:
            # now, reload with no chunking/dask and do the regridding
            subsampled_data = xr.open_dataset(filename_out)
        
        
        
        
        # get times associated with profiles:
        if SAMPLING_STRATEGY == 'sim_mooring':
            # - for mooring, use the subsampled time grid:
            times = np.unique(subsampled_data.time.values)
        else:
            # -- for glider/uctd, take the shallowest & deepest profiles (every second value, since top/bottom get sampled twice for each profile)
            time_deepest = subsampled_data.time.where(subsampled_data.dep == subsampled_data.dep.min(), drop=True).values[0:-1:2]
            time_shallowest = subsampled_data.time.where(subsampled_data.dep == subsampled_data.dep.max(), drop=True).values[0:-1:2]
            times = np.sort(np.concatenate((time_shallowest, time_deepest)))
            # this results in a time grid that may not be uniformly spaced, but is correct
            # - for a uniform grid, use the mean time spacing - may not be perfectly accurate, but is evenly spaced
            dt = np.mean(np.diff(time_shallowest))/2 # average spacing of profiles (half of one up/down, so divide by two)
            times_uniform = np.arange(survey_track.n_profiles.values*2) * dt

        # nt is the number of profiles (times):
        nt = len(times)  
        # xgr is the vertical grid; nz is the number of depths for each profile
        # depths are negative, so sort in reverse order using flip
        zgridded = np.flip(np.unique(subsampled_data.dep.data))
        nz = int(len(zgridded))

        # -- initialize the dataset:
        sgridded = xr.Dataset(
            coords = dict(depth=(["depth"],zgridded),
                      time=(["time"],times))
        )
        # -- 3-d fields: loop & reshape 3-d data from profiles to a 2-d (depth-time) grid:
        # first, extract each variable, then reshape to a grid
        
        for vbl in vbls3d:
            print(vbl)
            if sampling_details['SAVE_PRELIMINARY']:
                # not a dask array, so no "compute" command needed
                this_var = subsampled_data[vbl].data.copy() 
            else:
                this_var = subsampled_data[vbl].data.compute().copy() 
            # reshape to nz,nt
            this_var_reshape = np.reshape(this_var,(nz,nt), order='F') # fortran order is important!
            # for platforms with up & down profiles (uCTD and glider),
            # every second column is upside-down (upcast data)
            # starting with the first column, flip the data upside down so that upcasts go from top to bottom
            if SAMPLING_STRATEGY != 'sim_mooring':
                this_var_fix = this_var_reshape.copy()
                #this_var_fix[:,0::2] = this_var_fix[-1::-1,0::2] 
                this_var_fix[:,1::2] = this_var_fix[-1::-1,1::2]  # Starting with SECOND column
                sgridded[vbl] = (("depth","time"), this_var_fix)
            elif SAMPLING_STRATEGY == 'sim_mooring':
                sgridded[vbl] = (("depth","time"), this_var_reshape)
                
                
        if sampling_details['DERIVED_VARIABLES']:
            # for sampled steric height, we want the value integrated from the deepest sampling depth:
            sgridded['steric_height'] = (("time"), sgridded['steric_height'].isel(depth=nz-1).data)
            # rename to "steric_height_sampled" for clarity
            sgridded.rename_vars({'steric_height':'steric_height_sampled'})

  

        #  -- 2-d fields: loop & reshape 2-d data to the same time grid 
        for vbl in vbls2d:
            
            if sampling_details['SAVE_PRELIMINARY']:
                # not a dask array, so no "compute" command needed
                this_var = subsampled_data[vbl].data.copy() 
            else:
                this_var = subsampled_data[vbl].data.compute().copy() 
            # subsample to nt
            this_var_sub = this_var[0:-1:nz]
            sgridded[vbl] = (("time"), this_var_sub)

    # ------------ RETURN INTERPOLATED & GRIDDED DATA ------------

    # -- add variable attributes from ds
    if SAMPLING_STRATEGY == 'mooring':
        sgridded.attrs = ds.attrs
    else:
        # - find which variables in ds are also in our interpolated dataset:
        vars_ds = list(ds.keys())
        vars_sdata = list(subsampled_data.keys())
        vars_both = list(set(vars_ds) & set(vars_sdata))
        for var in vars_both:
            # copy over the attribute from ds:
            subsampled_data[var].attrs = ds[var].attrs
            sgridded[var].attrs = ds[var].attrs
    
    
    
    return subsampled_data, sgridded


# great circle distance (from Jake Steinberg) 
def great_circle(lon1, lat1, lon2, lat2):
    """Interpolates dataset 'ds' along the survey track given by the sruvey coordinates.


    Args:
        ds (xarray.core.dataset.Dataset): MITgcm LLC4320 data for all days
        survey_track (xarray.core.dataset.Dataset): lat,lon,dep,time of the survey used for the interpolation
        survey_indices (xarray.core.dataset.Dataset): i,j,k coordinates used for the interpolation
        sampling_details (dict):Includes number of days, waypoints, and depth range, horizontal and vertical platform speed. These can typical (default) or user-specified, in the                                      case where user specfies only some of the details the default values will be used for rest.
        

    Returns:
        subsampled_data: all field interpolated onto the track
        sh_true: 'true' steric height along the track
        
    Raises: 
        Sampling strategy is invalid: If a sampling strategy is not specified or different from the available strategies - sim_utcd, sim_glider, sim_mooring, wave_glider, sail_drone
    

    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))


# --------------------------------------------------------------------
# USER INPUTS:
# --------------------------------------------------------------------

# specify region from this list:
# WesternMed  ROAM_MIZ  NewCaledonia  NWPacific  BassStrait  RockallTrough  ACC_SMST
# MarmaraSea  LabradorSea  CapeBasin
RegionName = 'ACC_SMST' 

# specify date range as start date & number of days.
start_date = date(2012,1,1)
# NOTE: ndays must be >1 
ndays = 90

# directory where data files are stored

# kd: lab
#datadir = '/Users/kdrushka/data/adac/mitgcm/netcdf/' + RegionName + '/'                      # input model data are here
#outputdir = '/Users/kdrushka/data/adac/osse_output/' + RegionName + '/'           # interpolated data stored here
#figdir = '/Users/kdrushka/Dropbox/projects/adac/figures/' + RegionName + '/' # store figures

#mb: laptop
#datadir = '/Volumes/TOSHIBA EXT 1/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_model_input'  # input model data are here
#outputdir = '/Volumes/TOSHIBA EXT 1/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_output'   # interpolated data stored here
#figdir = '/Volumes/TOSHIBA EXT 1/LLC4320_pre-SWOT_10_days/ACC_SMST/figures' # store figures

#ocean computer
#datadir = '/mnt/data/CapeBasin/osse_model_input/30days'  # input model data are here
#outputdir = '/mnt/data/CapeBasin/osse_output'   # interpolated data stored here
#figdir = '/mnt/data/CapeBasin/figures' # store figures#datadir = '/home/manjaree/Documents/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_model_input/'  # input model data are here

#ocean computer
datadir = '/mnt/data/ACC_SMST/osse_model_input/30days/90days'  # input model data are here for 120 days
#datadir = '/mnt/data/ACC_SMST/osse_model_input/10days'  # input model data are here for 10 days
outputdir = '/mnt/data/ACC_SMST/osse_output'   # interpolated data stored here
figdir = '/mnt/data/ACC_SMST/figures' # store figures#datadir = '/home/manjaree/Documents/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_model_input/'  # input model data are here

 
 #datadir = '/mnt/data/ACC_SMST/osse_model_input/30days'  # input model data are here
#outputdir = '/mnt/data/ACC_SMST/osse_output'   # interpolated data stored here
#figdir = '/mnt/data/ACC_SMST/figures' # store figures#datadir = '/home/manjaree/Documents/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_model_input/'  # input model data are here

#outputdir = '/home/manjaree/Documents/LLC4320_pre-SWOT_10_days/ACC_SMST/osse_output'   # interpolated data stored here
#figdir = '/home/manjaree/Documents/LLC4320_pre-SWOT_10_days/ACC_SMST/figures' # store figures

SAVE_FIGURES = False # True or False


# optional details for sampling (if not specified, reasonable defaults will be used)
# NOTE!! mooring and sim_mooring are different:
#    sim_mooring treats the mooring datapoints like a glider, 
#    whereas mooring interpolates directly to the mooring grid and should be faster

sampling_details_sim_glider = {
  'SAMPLING_STRATEGY' : 'sim_glider', 
#   'SAMPLING_STRATEGY' : 'trajectory_file', # options: sim_glider, sim_uctd, wave_glider or trajectory_file.add:  ASV
#   'SAMPLING_STRATEGY' : 'mooring', # options: sim_glider, sim_uctd, sim_mooring or trajectory_file.add: ASV. 
#    'SAMPLING_STRATEGY' : 'wave_glider', # options: wave_glider ..add trajectory file too?
    'PATTERN' : 'lawnmower', # back-forth or lawnmower 
   'zrange' : [-1, -1000],  # depth range of T/S profiles (down is negative). * add U/V range? *
#   'zmooring_TS' : list(range(-10,-1000,-10)) # instrument depths for moorings. T/S and U/V are the same.
 #   'zrange' : [-6, -100], # instrument depths for wave glider.  
  'hspeed' : 0.25,  # platform horizontal speed in m/s (for glider, uCTD)
 #   'hspeed' : 1,# platform horizontal (profile) speed in m/s  (for wave_glider)
   'vspeed' : 0.1, # platform vertical (profile) speed in m/s  (for glider, uCTD)
  # 'vspeed': 0 ,# platform vertical (profile) speed in m/s  (for wave_glider)
    'trajectory_file' : '../data/survey_trajectory_ACC_SMST_glider.nc', # if SAMPLING_STRATEGY = 'trajectory_file', specify trajectory file
    'AT_END' : 'reverse', # behaviour at and of trajectory: 'reverse', 'repeat' or 'terminate'. (could also 'restart'?)
    'DERIVED_VARIABLES' : False, # specify whether or not to process the derived variables (steric height, rotated velocity, vorticity) - slower and takes significant to derive/save the stored variables
  }

sampling_details_mooring = {
#  'SAMPLING_STRATEGY' : 'sim_glider', 
#   'SAMPLING_STRATEGY' : 'trajectory_file', # options: sim_glider, sim_uctd, wave_glider or trajectory_file.add:  ASV
   'SAMPLING_STRATEGY' : 'mooring', # options: sim_glider, sim_uctd, sim_mooring or trajectory_file.add: ASV. 
#    'SAMPLING_STRATEGY' : 'wave_glider', # options: wave_glider ..add trajectory file too?
    'PATTERN' : 'lawnmower', # back-forth or lawnmower 
#   'zrange' : [-1, -1000],  # depth range of T/S profiles (down is negative). * add U/V range? *
  'zmooring_TS' : list(range(-10,-1000,-10)), # instrument depths for moorings. T/S and U/V are the same.
 #   'zrange' : [-6, -100], # instrument depths for wave glider.  
 # 'hspeed' : 0.25,  # platform horizontal speed in m/s (for glider, uCTD)
 #   'hspeed' : 1,# platform horizontal (profile) speed in m/s  (for wave_glider)
#   'vspeed' : 0.1, # platform vertical (profile) speed in m/s  (for glider, uCTD)
  # 'vspeed': 0 ,# platform vertical (profile) speed in m/s  (for wave_glider)
    'trajectory_file' : '../data/survey_trajectory_ACC_SMST_glider.nc', # if SAMPLING_STRATEGY = 'trajectory_file', specify trajectory file
#    'AT_END' : 'reverse', # behaviour at and of trajectory: 'reverse', 'repeat' or 'terminate'. (could also 'restart'?)
    'DERIVED_VARIABLES' : False, # specify whether or not to process the derived variables (steric height, rotated velocity, vorticity) - slower and takes significant to derive/save the stored variables
  }

sampling_details_wave_glider = {
  'SAMPLING_STRATEGY' : 'sim_glider', 
#   'SAMPLING_STRATEGY' : 'trajectory_file', # options: sim_glider, sim_uctd, wave_glider or trajectory_file.add:  ASV
#   'SAMPLING_STRATEGY' : 'mooring', # options: sim_glider, sim_uctd, sim_mooring or trajectory_file.add: ASV. 
 #   'SAMPLING_STRATEGY' : 'wave_glider', # options: wave_glider ..add trajectory file too?
    'PATTERN' : 'back-forth', # back-forth or lawnmower 
 #  'zrange' : [-1, -1000],  # depth range of T/S profiles (down is negative). * add U/V range? *
#   'zmooring_TS' : list(range(-10,-1000,-10)) # instrument depths for moorings. T/S and U/V are the same.
 #   'zrange' : [-6, -100], # instrument depths for wave glider.  
 # 'hspeed' : 0.25,  # platform horizontal speed in m/s (for glider, uCTD)
 #   'hspeed' : 1,# platform horizontal (profile) speed in m/s  (for wave_glider)
  # 'vspeed' : 0.1, # platform vertical (profile) speed in m/s  (for glider, uCTD)
 #  'vspeed': 0.0001 ,# platform vertical (profile) speed in m/s  (for wave_glider)
    'trajectory_file' : '../data/survey_trajectory_ACC_SMST_glider.nc', # if SAMPLING_STRATEGY = 'trajectory_file', specify trajectory file
  #  'AT_END' : 'terminate', # behaviour at and of trajectory: 'reverse', 'repeat' or 'terminate'. (could also 'restart'?)
    'DERIVED_VARIABLES' : False # specify whether or not to process the derived variables (steric height, rotated velocity, vorticity) - slower and takes significant to derive/save the stored variables
}

#    CONTROLS  
#sampling_details = sampling_details_mooring
sampling_details = sampling_details_sim_glider
#sampling_details = sampling_details_wave_glider
sampling_details

# download files:
#download_llc4320_data(RegionName, datadir, start_date, ndays)

# Trying without compute derived fields
# derive & save new files with steric height & vorticity
#if sampling_details['DERIVED_VARIABLES']:
 #  compute_derived_fields1(RegionName, datadir, start_date, ndays)
    

# Load all model data files
date_list = [start_date + timedelta(days=x) for x in range(ndays)]
#target_files = [f'{datadir}_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list target files
target_files = [f'{datadir}/LLC4320_pre-SWOT_ACC_SMST_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # lis
# chunk size ... aiming for ~100 MB chunks
# these chunks seem to work OK for up to ~20 day simulations, but more 
# testing is needed to figure out optimal parameters for longer simulations
#tchunk = 6 
#xchunk = 200
#ychunk = 200

#original
#tchunk = 6 
#xchunk = 150
#ychunk = 150

# drop the vector variables if loading derived variables because we are going to load the rotated ones in the next cell
#if sampling_details['DERIVED_VARIABLES']:
 #   drop_variables={'U', 'V', 'oceTAUX', 'oceTAUY'}
#else:
 #   drop_variables={}

ds = xr.open_mfdataset(target_files, 
                       #parallel=True, 
                       #drop_variables=drop_variables,
                      #chunks={'i':xchunk, 'j':ychunk, 'time':tchunk}
                      )

# XC, YC and Z are the same at all times, so select a single time
# (note, this breaks for a single file - always load >1 file)
X = ds.XC.isel(time=0) 
Y = ds.YC.isel(time=0)


# load the corresponding derived fields (includes steric height, vorticity, and transformed vector variables for current and wind stress)
if sampling_details['DERIVED_VARIABLES']:
    derivedir = datadir + 'derived/'
    derived_files = [f'{derivedir}LLC4320_pre-SWOT_{RegionName}_derived-fields_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list target files
    dsd = xr.open_mfdataset(derived_files, parallel=True, chunks={'i':xchunk, 'j':ychunk, 'time':tchunk})
    
    # merge the derived and raw data
    ds = ds.merge(dsd)
    # rename the transformed vector variables to their original names
    ds = ds.rename_vars({'U_transformed':'U', 'V_transformed':'V', 
                         'oceTAUX_transformed':'oceTAUX', 'oceTAUY_transformed':'oceTAUY'})


# drop a bunch of other vars we don't actually use - can comment this out if these are wanted
ds = ds.drop_vars({'DXV','DYU', 'DXC','DXG', 'DYC','DYG', 'XC_bnds', 'YC_bnds', 'Zp1', 'Zu','Zl','Z_bnds', 'nb'})


#del sys.modules['osse_tools_Copy1'] 
#from osse_tools_Copy1 import download_llc4320_data, compute_derived_fields, get_survey_track, survey_interp

survey_track, survey_indices, sampling_parameters = get_survey_track(ds, sampling_details)

# print specified sampling_details + any default values
print(sampling_parameters)


# ---- generate name of file to save outputs in ---- 
filename_base = (f'OSSE_{RegionName}_{sampling_details["SAMPLING_STRATEGY"]}_{start_date}_to_{start_date + timedelta(ndays)}_maxdepth{int(sampling_parameters["zrange"][1])}')
filename_out_base = (f'{outputdir}/{filename_base}')


# Visualise the track over a single model snappshot
plt.figure(figsize=(15,5))

# map of Theta at time zero
ax = plt.subplot(1,2,1)
ssto = plt.pcolormesh(X,Y,ds.Theta.isel(k=0, time=0).values, shading='auto')
if not (sampling_details['SAMPLING_STRATEGY'] == 'mooring' or sampling_details['SAMPLING_STRATEGY'] == 'sim_mooring'):
    tracko = plt.scatter(survey_track.lon, survey_track.lat, c=(survey_track.time-survey_track.time[0])/1e9/86400, cmap='Reds', s=0.75)
    plt.colorbar(ssto).set_label('SST, $^o$C')
    plt.colorbar(tracko).set_label('days from start')
    plt.title('SST and survey track: ' + RegionName + ', '+ sampling_details['SAMPLING_STRATEGY'])
else:
    plt.plot(survey_track.lon, survey_track.lat, marker='*', c='r')
    plt.title('SST and mooring location: ' + RegionName + ' region, ' + sampling_details['SAMPLING_STRATEGY'] )


# depth/time plot of first few datapoints
ax = plt.subplot(1,2,2)
iplot = slice(0,20000)
if not (sampling_details['SAMPLING_STRATEGY'] == 'mooring' or sampling_details['SAMPLING_STRATEGY'] == 'sim_mooring'):
    plt.plot(survey_track.time.isel(points=iplot), survey_track.dep.isel(points=iplot), marker='.')
else:
    # not quite right but good enough for now.
    # (times shouldn't increase with depth)
    plt.scatter((np.tile(survey_track['time'].isel(time=iplot), int(survey_track['dep'].data.size))),
         np.tile(survey_track['dep'], int(survey_track['time'].isel(time=iplot).data.size)),marker='.')             
#plt.xlim([start_date + datetime.timedelta(days=0), start_date + datetime.timedelta(days=2)])
plt.ylabel('Depth, m')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gcf().autofmt_xdate()
plt.title(f"Sampling pattern, hspeed ={sampling_parameters['hspeed']}, vspeed ={sampling_parameters['vspeed']}")


# save
if SAVE_FIGURES:
    #plt.savefig('/data2/Dropbox/projects/adac/figures/' + filename_base + '_sampling.png', dpi=400, transparent=False, facecolor='white')
    plt.savefig(figdir + '/' + filename_base + '_sampling.png', dpi=400, transparent=False, facecolor='white')

plt.show()


print (sampling_parameters)


#Interpolate with the specified pattern (where the magic happens)
#del sys.modules['osse_tools'] 
#from osse_tools import survey_interp, get_survey_track

subsampled_data, sgridded = survey_interp(ds, survey_track, survey_indices, sampling_parameters)
sgridded



# 3d fields
vbls3d = ['Theta','Salt','U','V','vorticity']
vbls3d = ['Theta','Salt']
ylim = [min(sgridded['depth'].values), max(sgridded['depth'].values)]
ylim = [-200, -1]

nr = len(vbls3d) # # of rows
fig,ax=plt.subplots(nr,figsize=(8,len(vbls3d)*2),constrained_layout=True)


for j in range(nr):
    sgridded[vbls3d[j]].plot(ax=ax[j], ylim=ylim)
    ax[j].plot(sgridded.time.data, -sgridded.KPPhbl.data, c='k')
    ax[j].set_title(vbls3d[j])

if SAVE_FIGURES:
    plt.savefig(figdir + '/' + filename_base + '_3D.png', dpi=400, transparent=False, facecolor='white')
    