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

def setup_earthdata_login_auth(endpoint: str='urs.earthdata.nasa.gov'):
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
    
    
def download_llc4320_data_demo(RegionName, datadir, start_date, ndays):
    """
    Check for existing llc4320 files in 'datadir' and download if they aren't found
    inputs XXX
    """
    ShortName = "MITgcm_LLC4320_Pre-SWOT_JPL_L4_" + RegionName + "_v1.0"
    date_list = [start_date + datetime.timedelta(days=x) for x in range(ndays)]
    target_files = [f'LLC4320_pre-SWOT_{RegionName}_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list of files to check for/download
    setup_earthdata_login_auth()
    
    # https access for each target_file
    url = "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected"
    https_accesses = [f"{url}/{ShortName}/{target_file}" for target_file in target_files]
#     print(https_accesses)
    

#     def begin_s3_direct_access():
#     """Returns s3fs object for accessing datasets stored in S3."""
#     response = requests.get("https://archive.podaac.earthdata.nasa.gov/s3credentials").json()
#     return s3fs.S3FileSystem(key=response['accessKeyId'],
#                              secret=response['secretAccessKey'],
#                              token=response['sessionToken'], 
#                              client_kwargs={'region_name':'us-west-2'})

    # list of dataset objects
    dds = []
    for https_access,target_file in zip(https_accesses,target_files):
        print(target_file) # print file name

        if not(os.path.isfile(datadir + target_file)):
            filename_dir = os.path.join(datadir, target_file)
            request.urlretrieve(https_access, filename_dir)
           
            
def compute_derived_fields_demo(RegionName, datadir, start_date, ndays):
    """
    Check for derived files in 'datadir'/derived and compute if the files don't exist
    """
    # directory to save derived data to - create if doesn't exist
    derivedir = datadir + 'derived/'
    if not(os.path.isdir(derivedir)):
        os.mkdir(derivedir)
        
    # files to load:
    date_list = [start_date + datetime.timedelta(days=x) for x in range(ndays)]
    target_files = [f'{datadir}LLC4320_pre-SWOT_{RegionName}_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list target files
    
    # list of derived files:
    derived_files = [f'{derivedir}LLC4320_pre-SWOT_{RegionName}_derived-fields_{date_list[n].strftime("%Y%m%d")}.nc' for n in range(ndays)] # list target files

        
    # loop through input files, then compute steric height, vorticity, etc. on the i/j grid
    fis = range(len(target_files))
    
    cnt = 0 # count
    for fi in fis:
        # input filename:
        thisf=target_files[fi]
        # output filename:
        fnout = thisf.replace(RegionName + '_' , RegionName + '_derived-fields_')
        fnout = fnout.replace(RegionName + '/' , RegionName + '/derived/')
        # check if output file already exists
        if (not(os.path.isfile(fnout))):   
            print('computing derived fields for', thisf) 
            # load file:
            ds = xr.open_dataset(thisf)
            
            # -------
            # first time through the loop, load reference profile:
            # load a single file to get coordinates
            if cnt==0:
                # mean lat/lon of domain
                xav = ds.XC.isel(j=0).mean(dim='i')
                yav = ds.YC.isel(i=0).mean(dim='j')

                # for vorticity calculation, build the xgcm grid:
                # see https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
                grid = xgcm.Grid(ds, coords={'X':{'center': 'i', 'left': 'i_g'}, 
                             'Y':{'center': 'j', 'left': 'j_g'},
                             'T':{'center': 'time'},
                             'Z':{'center': 'k'}})

                # load reference file of argo data
                # NOTE: could update to pull from ERDDAP or similar
                argoclimfile = '/data1/argo/argo_CLIM_3x3.nc'
                argods = xr.open_dataset(argoclimfile,decode_times=False) 
                # reference profiles: annual average Argo T/S using nearest neighbor
                Tref = argods["TEMP"].sel(LATITUDE=yav,LONGITUDE=xav, method='nearest').mean(dim='TIME')
                Sref = argods["SALT"].sel(LATITUDE=yav,LONGITUDE=xav, method='nearest').mean(dim='TIME')
                # SA and CT from gsw:
                # see example from https://discourse.pangeo.io/t/wrapped-for-dask-teos-10-gibbs-seawater-gsw-oceanographic-toolbox/466
                Pref = xr.apply_ufunc(sw.p_from_z, -argods.LEVEL, yav)
                Pref.compute()
                SAref = xr.apply_ufunc(sw.SA_from_SP, Sref, Pref, xav, yav,
                                       dask='parallelized', output_dtypes=[Sref.dtype])
                SAref.compute()
                CTref = xr.apply_ufunc(sw.CT_from_pt, Sref, Tref, # Theta is potential temperature
                                       dask='parallelized', output_dtypes=[Sref.dtype])
                CTref.compute()
                Dref = xr.apply_ufunc(sw.density.rho, SAref, CTref, Pref,
                                    dask='parallelized', output_dtypes=[Sref.dtype])
                Dref.compute()
                cnt = cnt+1
                print()
            # -------
            # 
            # --- compute steric height in steps ---
            # 0. create datasets for variables of interest:
            ss = ds.Salt
            tt = ds.Theta
            pp = xr.DataArray(sw.p_from_z(ds.Z,ds.YC))
            
            # 1. compute absolute salinity and conservative temperature
            sa = xr.apply_ufunc(sw.SA_from_SP, ss, pp, xav, yav, dask='parallelized', output_dtypes=[ss.dtype])
            sa.compute()
            ct = xr.apply_ufunc(sw.CT_from_pt, sa, tt, dask='parallelized', output_dtypes=[ss.dtype])
            ct.compute()
            dd = xr.apply_ufunc(sw.density.rho, sa, ct, pp, dask='parallelized', output_dtypes=[ss.dtype])
            dd.compute()
            # 2. compute specific volume anomaly: gsw.density.specvol_anom_standard(SA, CT, p)
            sva = xr.apply_ufunc(sw.density.specvol_anom_standard, sa, ct, pp, dask='parallelized', output_dtypes=[ss.dtype])
            sva.compute()
            # 3. compute steric height = integral(0:z1) of Dref(z)*sva(z)*dz(z)
            # - first, interpolate Dref to the model pressure levels
            Drefi = Dref.interp(LEVEL=-ds.Z)
            dz = -ds.Z_bnds.diff(dim='nb').drop_vars('nb').squeeze() # distance between interfaces

            # steric height computation (summation/integral)
            # - increase the size of Drefi and dz to match the size of sva
            Db = Drefi.broadcast_like(sva)
            dzb = dz.broadcast_like(sva)
            dum = Db * sva * dzb
            sh = dum.cumsum(dim='k')

            # --- compute vorticity using xgcm and interpolate to X, Y
            # see https://xgcm.readthedocs.io/en/latest/xgcm-examples/02_mitgcm.html
            vorticity = (grid.diff(ds.V*ds.DXG, 'X') - grid.diff(ds.U*ds.DYG, 'Y'))/ds.RAZ
            vorticity = grid.interp(grid.interp(vorticity, 'X', boundary='extend'), 'Y', boundary='extend')

            # --- save derived fields in a new file
            # - convert sh and zeta to datasets
            dout = vorticity.to_dataset(name='vorticity')
            sh_ds = sh.to_dataset(name='steric_height')
            dout = dout.merge(sh_ds)
            # add/rename the Argo reference profile variables
            tref = Tref.to_dataset(name='Tref')
            tref = tref.merge(Sref).rename({'SALT': 'Sref'}).\
                rename({'LEVEL':'zref','LATITUDE':'yav','LONGITUDE':'xav'}).\
                drop_vars({'i','j'})
            # - add ref profiles to dout and drop uneeded vars/coords
            dout = dout.merge(tref).drop_vars({'LONGITUDE','LATITUDE','LEVEL','i','j'})

            # - save netcdf file with derived fields
            netcdf_fill_value = nc4.default_fillvals['f4']
            dv_encoding = {}
            for dv in dout.data_vars:
                dv_encoding[dv]={'zlib':True,  # turns compression on\
                            'complevel':9,     # 1 = fastest, lowest compression; 9=slowest, highest compression \
                            'shuffle':True,    # shuffle filter can significantly improve compression ratios, and is on by default \
                            'dtype':'float32',\
                            '_FillValue':netcdf_fill_value}
            # save to a new file
            print(' ... saving to ', fnout)
            dout.to_netcdf(fnout,format='netcdf4',encoding=dv_encoding)


# great circle distance (from Jake Steinberg) 
def great_circle(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
