import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DateFormatter
from pylab import*
c = 299792458
dtr = np.pi/180

def dop_df(file_path):
    """
    Load DOP data from a file and process it.

    Args:
        file_path (str): Path to the DOP data file.

    Returns:
        pd.DataFrame: Processed DOP data with calculated TDOP.
    """
    df = pd.read_fwf(file_path)
    df.columns = ['Epoch', 'NSAT', 'GDOP', 'PDOP', 'HDOP', 'VDOP', 'EL']
    df["TDOP"] = np.sqrt(df["GDOP"] ** 2 - df["PDOP"] ** 2)
    df["Epoch_datetime"] = pd.to_datetime(df['Epoch'], format='%Y/%m/%d %H:%M:%S.%f')
    return df


def plot_nsat(file_paths, labels, colors, output_path=None):
    """
    Plot the number of satellites (NSAT) over time for multiple GNSS systems.

    Args:
        file_paths (list of str): Paths to the input DOP data files.
        labels (list of str): Labels for each dataset to appear in the legend.
        colors (list of str): Colors for each dataset's plot line.
        output_path (str, optional): Path to save the output plot. If None, the plot is displayed.
    """
    date_form = DateFormatter("%H")
    fig, ax = plt.subplots(figsize=(9, 12), dpi=600)
    
    for file_path, label, color in zip(file_paths, labels, colors):
        df = dop_df(file_path)
        date_str = df["Epoch"].iloc[15].split(" ")[0]
        date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d")
        formatted_date = date_obj.strftime("UTC %d %b %Y")
        
        ax.plot(df["Epoch_datetime"], df["NSAT"], color=color, label=label, lw=3)
    
    ax.xaxis.set_major_formatter(date_form)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_yticks(np.arange(0, 55, 5))
    ax.set_xlabel(formatted_date, fontsize=24)
    ax.set_ylabel("Number of Satellites", fontsize=24)
    ax.legend(ncol=2, loc="upper right", fontsize=18)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path+"nsat.png")
    else:
        plt.show()


def plot_dop(file_paths, labels, colors, output_path=None):
    """
    Plot DOP components (GDOP, PDOP, HDOP, VDOP, TDOP) for multiple GNSS systems.

    Args:
        file_paths (list of str): Paths to the input DOP data files.
        labels (list of str): Labels for the GNSS systems.
        colors (list of str): Colors for DOP components.
        output_path (str, optional): Path to save the output plot. If None, the plot is displayed.
    """
    date_form = DateFormatter("%H")
    fig, ax = plt.subplots(nrows=5, ncols=1, sharey="row", sharex="col", figsize=(9, 14), dpi=600)

    dop_data = [dop_df(path) for path in file_paths]
    
    for i, (df, label) in enumerate(zip(dop_data[::-1], labels[::-1])):
        ax[i].plot(df["Epoch_datetime"], df["GDOP"], color=colors[0], label="GDOP")
        ax[i].plot(df["Epoch_datetime"], df["PDOP"], color=colors[1], label="PDOP")
        ax[i].plot(df["Epoch_datetime"], df["HDOP"], color=colors[2], label="HDOP")
        ax[i].plot(df["Epoch_datetime"], df["VDOP"], color=colors[3], label="VDOP")
        ax[i].plot(df["Epoch_datetime"], df["TDOP"], color=colors[4], label="TDOP")
        
        ax[i].xaxis.set_major_formatter(date_form)
        ax[i].grid(True)
        ax[i].set_ylim([0, 6])
        ax[i].tick_params(axis='both', which='major', labelsize=24)
        ax[i].set_yticks(np.arange(0, 7, 1))
        ax[i].set_ylabel(label, fontsize=24)

        if i == 0:
            ax[i].legend(ncol=3, loc="upper right", fontsize=18)

    ax[-1].set_xlabel("UTC 15 Oct 2022", fontsize=24)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path+"dop.png")
    else:
        plt.show()



def cart2ellipsoid(arr_x):
    
    '''
    

    Parameters
    ----------
    arr_x : Array
        Array containing cartesian coordinate in which 
        arr_x = [[x],[y],[z]]

    Returns
    -------
    arr_ep : Array containing ellipsoidal coordinate(phi,lamda,height)
        phi     = arr_ep[0][0]
        lambda  = arr_ep[1][0]
        h       = arr_ep[2][0]
    '''
    
    x = arr_x[0][0]
    y = arr_x[1][0]
    z = arr_x[2][0]

    
    # WGS84 ellipsoid constants:
    a = 6378137
    f = 1/298.257
    e = np.sqrt(2*f-f**2)
    
    #calculations:
    lmda = np.arctan2(y,x)
    
    p   = np.sqrt(x**2+y**2)
    
    h = 0
    phi = np.arctan2(z,p*(1-e**2))
    
    N = a/(1-(e*np.sin(phi))**2)**0.5
    
    delta_h = 1000000
    while delta_h > 0.01:
        prev_h = h
        phi = np.arctan2(z,p*(1-e**2*(N/(N+h))))
        N = a/(1-(e*np.sin(phi))**2)**0.5
        h = p/np.cos(phi) - N
        delta_h = abs(h - prev_h)
    
    b   = np.sqrt((a**2)*(1-e**2))
    ep  = np.sqrt((a**2-b**2)/b**2)
    a11 = phi
    a21 = lmda
    a31 = h
    
    arr_ep = [[a11],[a21],[h]]
    
    arr_epp = np.array(arr_ep)
    

    return arr_ep

def antennapos(file_observation):
    
    '''
    antennapos fetches antenna position from rinex3 observation file

    Input = filepath of Rinex3 Observaion File

    Output = list containing cartesian coordinates of antenna in ECEF
    '''
    
    df = pd.DataFrame()
    X = []
    Y = []
    Z = []
    
    with open(file_observation) as f:
        for i, line in enumerate(f):
            if 'APPROX POSITION XYZ' in line:
                '''Geocentric approximate marker position (Units: Meters,
                System: ITRS recommended) 
                Optional for moving platforms'''
            
                approx_marker_position_X = float(line[0:14].strip())
                X.append(approx_marker_position_X)
                approx_marker_position_Y = float(line[14:28].strip())
                Y.append(approx_marker_position_Y)
                approx_marker_position_Z = float(line[28:42].strip())
                Z.append(approx_marker_position_Z)
    arr_x = np.array([[X],[Y],[Z]])
    
    phi_u   = cart2ellipsoid(arr_x)[0][0]
    lmbda_u = cart2ellipsoid(arr_x)[1][0]
    h_u     = cart2ellipsoid(arr_x)[2][0]
    df['X'] = X
    df['Y'] = Y
    df['Z'] = Z
    df['lat_u'] = phi_u
    df['long_u'] = lmbda_u
    df['h_u'] = h_u
    return df

def observationtype(filepath):
    '''
    Parameters
    ----------
    filepath : String
        filepath of Rinex3 observation file

    Returns
    -------
    df_obsv : Pandas DataFrae
        Dataframe containing Satellite system as Index and number of 
        observation and type of information as another column in the DataFrame.
    '''
    #empty lists and string
    observation = []
    obs=""
    Satelite_system = []
    number_of_obs   = []
    type_of_obsv = []

    with open(filepath) as f:
        for line in f.readlines():
            if 'OBS TYPES' in line:
                observation.append(line[0:60])

    for i in observation:
        obs += i

    obsv=list(obs.split(" "))
    for i in range(obsv.count('')):
        obsv.remove('')
    i=0
    while i <len(obsv):
        satid=obsv[i]
        Satelite_system.append(satid)
        no_of_obsv = int(obsv[i+1])
        number_of_obs.append(no_of_obsv)
        type_data=obsv[i+2:no_of_obsv+2+i]
        type_of_obsv.append(type_data)
        i=i+2+no_of_obsv

    df_obsv = pd.DataFrame()
    for i in range(len(Satelite_system)):

        dff =  pd.DataFrame([[Satelite_system[i],number_of_obs[i],type_of_obsv[i]]], columns=['Satelite_system','Number_of_observation','type_of_observation'])
        df_obsv = pd.concat([df_obsv, dff], ignore_index=True)
        
        #df_obsv = df_obsv.append(dff)
    df_obsv=df_obsv.set_index(['Satelite_system'])

    return df_obsv
def obsdata_filter(a):
        
    try:
        b = a.split(".")
        c = b[0].strip()+"."+b[1][0:3]
        c = float(c)
    except IndexError:
        c = np.nan
        
    
    return c

def obsdata_filter_cn(a):
    
    try:
        ab = a.strip()
       
        if len(ab)>8:
            try:
                b = float(a[-1:])
            except ValueError:
                b = np.nan
        else:
            b = np.nan
    except IndexError:
        b = np.nan
    
    
    return b
    



def epochdata_filter(ep_data):
    
    
    '''
    
    Convert Epoch data in string to a list containing year month day hour minute sec separately
    
    Parameters
    ----------
    ep_data : string
        epoch combine data

    Returns
    -------
    ep_list : List
        epoch separate data 
    '''
    ep_list = []
    ep_data = ep_data.split()
    epd_y = ep_data[0]
    ep_list.append(int(epd_y))
    epd_m = ep_data[1]
    if len(epd_m)==1:
        epd_m = '0'+epd_m
    ep_list.append(int(epd_m))
    epd_d = ep_data[2]
    
    if len(epd_d)==1:
        epd_d = '0'+epd_d
                
    ep_list.append(int(epd_d))
                
    epd_h = ep_data[3]
    if len(epd_h)==1:
        epd_h = '0'+epd_h
    ep_list.append(epd_h)
                
    epd_min = ep_data[4]
    if len(epd_min)==1:
        epd_min = '0'+epd_min
    ep_list.append(epd_min)
    epd_sec = ep_data[5][0:8]
    

    epd_sec1 = epd_sec.split(".")
    if len(epd_sec1[0]) ==1:
        epd_sec = "0"+epd_sec1[0] +"."+ epd_sec1[1]            
    #epd_sec = epd_sec.split(".")
    #epd_sec = epd_sec[0]
                
    if len(epd_sec)==1:
        epd_sec = '0'+epd_sec
    ep_list.append(epd_sec)
    
    ep_hour = float(epd_h) + float(epd_min)/60  + float(epd_sec)/3600 
    
    ep_list.append(ep_hour)
    epochd_flag = int(ep_data[6])
    
    epoch_f = epd_y+epd_m+epd_d+epd_h+epd_min+epd_sec
    
    
    
    ep_list.append(epoch_f)
    
    ep_list.append(epochd_flag)
    
    
    
    
    
    return ep_list

def gpstime(timestring):
    
    
    
    leapseconds = 0
    localoffset=0
    epoch = datetime.datetime.strptime("1980-01-07 00:00:00","%Y-%m-%d %H:%M:%S")
    timeformat = '%Y%m%d%H%M%S.%f'
    local = datetime.datetime.strptime(timestring,timeformat)
    utc = local - datetime.timedelta(hours=localoffset)
    diff = utc-epoch
    gpsWeek = diff.days/7
    secondsThroughDay = (utc.hour * 3600) + (utc.minute * 60) + utc.second
    if utc.isoweekday()== 7:
        weekday = 0
    else:
        weekday = utc.isoweekday()
    gpsSeconds = (weekday * 86400) + secondsThroughDay - leapseconds
    return gpsSeconds

def obsdataframe_rtk(f_obs,constellation):
    
    '''
    Parameters
    ----------
    f_obs : string
        file path of Rinex3 observation file        
    Returns
    -------
    df : DataFrame
        DataFrame Contains information of satellites observation Data        

    '''
    
    #arr_nav = gpsnav3df(f_navb)
    
    
    ###Approximate Antenna Position##
    df_antpos = antennapos(f_obs)
    xu = df_antpos.iloc[0]['X']
    yu = df_antpos.iloc[0]['Y']
    zu = df_antpos.iloc[0]['Z']
    
    #dataframe of observation type of all constellation
    
    df_obsv = observationtype(f_obs)
    
    #dataframe of observation type for specific constellation
    
    df_obsv_g = df_obsv.loc[constellation]
    
    list_obsv_g = df_obsv_g['type_of_observation'] #list
    list_obsv_cn = [x + "_CN" for x in list_obsv_g]
    

    with open(f_obs) as f:    
        f1 = f.read() 
        f2 = f1.split('END OF HEADER')
        f3 =f2[1]
        f4 = f3.split('>')
        f4 = f4[1:]
        epoch_list = [] #Epoch
        obs_data = []  #observation data
        cn_data  = [] #C/N data
        
        for i in f4:    
            #epoch
            ep_data = i.split('\n', 1)[0].strip()
            
            for j in i.splitlines()[1:]:
                s = 0
                if j[0] == constellation:
                    
                    #Satellite ID
                    satid = j[0:3].strip()
                    satnum = satid[1:3].strip()
                    satname = satid[0]
                    if len(satnum)==1:
                        satnum = "0"+satnum
                    satid = satname + satnum
                    
                    #Data
                    data_obs = j[3:]
                    data_obs = [data_obs[k:k+16] for k in range(0, len(data_obs), 16)]
                    
                    c1c = obsdata_filter(data_obs[0])
                    
                    
                    
                    if len(data_obs)!= len(list_obsv_g):
                        num_miss = len(list_obsv_g)-len(data_obs)
                        data_obs.extend(num_miss*[" "])
                    
                    
                    
                    
                    data_obs1 = map(obsdata_filter,data_obs)
                    obs_data.append(data_obs1)
                    
                    
                    data_cn  = map(obsdata_filter_cn,data_obs)
                    
                    
                    cn_data.append(data_cn)
                    
                    s = s+1
                    #Epoch
                    epoch1 = epochdata_filter(ep_data)
                    epoch1.insert(0,satid)
                    epoch_list.append(epoch1)
                    
                    
                    
                    if constellation == "G":
                    
                        g_time = gpstime(epoch1[-2])
                    
                        tt = g_time -c1c/c
                    
                        ep_datetime = datetime.datetime.strptime(epoch1[-2], '%Y%m%d%H%M%S.%f')
                    


                    
            
        
        df_data =  pd.DataFrame(obs_data, columns=list_obsv_g)
        
        df_cn   =  pd.DataFrame(cn_data, columns=list_obsv_cn)
        
        
    
        df_epoch = pd.DataFrame(epoch_list,columns=["SatId","Year","Month","Day","Hour","Minute","Sec","hour","Epoch","Epoch_flag"])
        df = pd.concat([df_epoch,df_data,df_cn], axis=1, ignore_index=False) #final dataframe
        a =  df['Epoch'].astype(float)
        a = a.round()
        df["Epoch"] = a
        df["Epoch"] = df["Epoch"].astype(str)

        
        
    return df
def obsdata_snr(file_obs,file_snr):
    
    '''
    Parameters
    ----------
    file_obs : string
        filepath of observation data file
    file_snr : string
        filepath of snr from rtklib
    
    Returns
    -------
    Observation Dataframe

    '''
    try:
        df_g = obsdataframe_rtk(file_obs,"G")
        df_g["constellation"] = "G"
        df_g["Epoch_datetime"] = pd.to_datetime(df_g['Epoch'],format = '%Y%m%d%H%M%S.%f')
    except KeyError:
        pass
    
    try:
        df_j = obsdataframe_rtk(file_obs,"J")
        df_j["constellation"] = "J"
        df_j["Epoch_datetime"] = pd.to_datetime(df_j['Epoch'],format = '%Y%m%d%H%M%S.%f')
    except KeyError:
        pass
    
    try:
        df_e = obsdataframe_rtk(file_obs,"E")
        df_e["constellation"] = "E"
        df_e["Epoch_datetime"] = pd.to_datetime(df_e['Epoch'],format = '%Y%m%d%H%M%S.%f')
        
    except KeyError:
        pass
    
    
    
    try:
        df_r = obsdataframe_rtk(file_obs,"R")
        df_r["constellation"] = "R"
        df_r["Epoch_datetime"] = pd.to_datetime(df_r['Epoch'],format = '%Y%m%d%H%M%S.%f')
    except KeyError:
        pass
    
    try:    
        df_c = obsdataframe_rtk(file_obs,"C")
        df_c["constellation"] = "C"
        df_c["Epoch_datetime"] = pd.to_datetime(df_c['Epoch'],format = '%Y%m%d%H%M%S.%f')
        
    except KeyError:
        pass
    
    try:
        df_i = obsdataframe_rtk(file_obs,"I")
        df_i["constellation"] = "I"
        df_i["Epoch_datetime"] = pd.to_datetime(df_i['Epoch'],format = '%Y%m%d%H%M%S.%f')
    except KeyError:
        pass
    
    
    try:
        df_obs11 = pd.concat([df_g, df_r,df_e,df_c,df_j,df_i], axis=0, join='outer')
    except UnboundLocalError:
        try: 
            df_obs11 = pd.concat([df_g, df_r,df_e,df_c,df_j], axis=0, join='outer')
        except UnboundLocalError :
            try:
                df_obs11 = pd.concat([df_g, df_r,df_e,df_j], axis=0, join='outer')
            except:
                df_obs11 = pd.concat([df_g, df_r,df_e], axis=0, join='outer')
            
            

    df_snr1 =  pd.read_fwf(file_snr)
    df_snr1.columns = ['Epoch', 'SatId','Azimuth',"Elevation","SNR","MP"]
    #print(df_snr1['Epoch'])
    
    
    
    df_snr1["Epoch_datetime"] = pd.to_datetime(df_snr1['Epoch'],format = '%Y/%m/%d %H:%M:%S.%f')
    df_snr1["Elev_Plot"] = 90-  df_snr1.Elevation

    df_obs = pd.merge(df_obs11, df_snr1,  how='left',on=['Epoch_datetime','SatId'])
    df_obs["Azimuth_r"] = df_obs.Azimuth*dtr    
    return df_obs


def plot_cnr(file_obs, file_snr,output_path=None,constellations=["G", "R", "E", "C"], cmap="plasma", snr_min=10, snr_max=55):
    """
    Generate polar scatter plots of CNR values for specified constellations from observation and SNR files.

    Parameters:
    -----------
    file_obs : str
        File path of the observation data (RINEX file).
    file_snr : str
        File path of the SNR data (from RTKLIB or similar).
    constellations : list, optional
        List of constellations to plot (default ["G", "R", "E", "C"]).
    cmap : str, optional
        Colormap for the SNR values (default "plasma").
    snr_min : int, optional
        Minimum SNR value for the colormap (default 10).
    snr_max : int, optional
        Maximum SNR value for the colormap (default 55).
    """

    # Process observation and SNR data
    df_obs = obsdata_snr(file_obs, file_snr)
    df_obs["Azimuth_r"] = df_obs.Azimuth * (np.pi / 180)  # Convert azimuth to radians
    df_obs["Elev_Plot"] = 90 - df_obs.Elevation          # Convert elevation to polar plot scale

    fig, axs = plt.subplots(len(constellations), 1, subplot_kw=dict(projection="polar"), figsize=(12, 12), dpi=600)

    if len(constellations) == 1:
        axs = [axs]  # Ensure axs is iterable for a single subplot

    for idx, constellation in enumerate(constellations):
        df_constellation = df_obs[df_obs["constellation"] == constellation]
        if df_constellation.empty:
            continue

        val = df_constellation["SNR"].values
        r = df_constellation['Elev_Plot'].values
        theta = df_constellation['Azimuth_r'].values

        scatter = axs[idx].scatter(theta, r, c=val, cmap=cmap, s=1, vmin=snr_min, vmax=snr_max)
        axs[idx].set_rmax(90)
        axs[idx].set_yticklabels([])
        axs[idx].grid(True)
        axs[idx].set_rorigin(0)
        axs[idx].set_theta_zero_location("N")
        axs[idx].set_theta_direction(-1)
        axs[idx].set_rticks([0, 15, 30, 45, 60, 75, 90])
        axs[idx].set_title(constellation, loc='left', fontsize=16, fontweight='bold')

    #fig.tight_layout(pad=1.5)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axs, ticks=np.arange(snr_min, snr_max + 5, 5),
                        orientation='vertical', location='left', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("L1 " + r"$\mathrm{C/N_0(db-Hz)}$", fontsize=14, rotation=270, labelpad=15)
    
    if output_path:
        fig.savefig(output_path+"cnr.png")
        plt.show()
    else:
        plt.show()

def plot_multipath(file_obs,file_mp,outputh_path = None,fontsize=24, vmax=1, vmin=-1):
    
    """
    Function to plot L1 Pseudorange Multipath as a polar plot.
    
    Parameters:
    df_look : DataFrame
        Data containing 'Azimuth_r', 'Elev_Plot', and 'MP' columns.
    fontsize : int, optional
        Font size for labels. Default is 24.
    vmax : float, optional
        Maximum colorbar limit. Default is 1.
    vmin : float, optional
        Minimum colorbar limit. Default is -1.
    """
    df_obs2 = obsdata_snr(file_obs,file_mp)
    df_obs2_g = df_obs2[df_obs2["constellation"]=="G"]
    df_look = df_obs2_g[["Epoch_datetime",'Elevation','Azimuth_r','Elev_Plot',"MP"]] 
    
    cmap = cm.get_cmap('RdGy', 8)
    fig, axs = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(16, 16), dpi=600)

    val = df_look["MP"].values
    r = df_look['Elev_Plot'].values
    theta = df_look['Azimuth_r'].values

    sc = axs.scatter(theta, r, c=val, cmap=cmap, s=10, vmin=vmin, vmax=vmax)
    axs.set_rmax(90)
    axs.set_yticklabels([])
    axs.set_rlabel_position(0)
    axs.grid(True)
    axs.set_rorigin(0)
    axs.set_theta_zero_location("N")
    axs.set_theta_direction(-1)
    axs.set_rticks([0, 15, 30, 45, 60, 75, 90])
    
    for i, text in enumerate(["75", "60", "45", "30", "15"]):
        axs.text(0, (i+1)*15, text, fontsize=fontsize)
    
    axs.tick_params(axis='x', which='major', labelsize=fontsize)
    
    # Move colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(sc, cax=cbar_ax, ticks=np.arange(vmin, vmax + .1, .25),
                        orientation='vertical', location='right', pad=0.2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_ylabel("L1 Pseudorange Multipath [m]", fontsize=fontsize, labelpad=20)
    
    if outputh_path:
        plt.savefig(outputh_path+"multipath.png")
        plt.show()
    else:
        plt.show()

    return df_look
def dph_data(file_path):
    Data = []
    column_names = [
        'Epoch', 'L1 cyc', 'L2 cyc', 'P1 cyc', 'P2 cyc', 'LC cyc', 
        'LG cyc', 'PC cyc', 'WL cyc', 'N cyc', 'LSV', 
        'Azimuth', 'Elev', 'PF', 'data_flag', 'L1_cycles', 'L2_cycles', 'PRN'
    ]
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                data = line.split()
                data = [np.nan if x == "********" else float(x) for x in data]
                Data.append(data)
        df = pd.DataFrame(Data, columns=column_names)
        if not df.empty:
            sat_id = df["PRN"].dropna().unique()
            df["L1_cycles"].replace(sat_id[0], np.nan, inplace=True)
            df["PRN"] = sat_id[0]
            df["Azimuth"] *= dtr
            df["Elev1"] = 90 - df["Elev"]
    except ValueError:
        column_names1 = [
            'Epoch', 'L1 cyc', 'L2 cyc', 'P1 cyc', 'P2 cyc', 'LC cyc', 
            'LG cyc', 'PC cyc', 'WL cyc', 'N cyc', 'LSV', 
            'Azimuth', 'Elev', 'PF', 'data_flag', 'PRN'
        ]
        df = pd.DataFrame(Data, columns=column_names1)
        if not df.empty:
            df["Azimuth"] *= dtr
            df["Elev1"] = 90 - df["Elev"]
    return df
def plot_lcphase(basepath,output_path=None):
    PRN = [f'PRN{i:02d}' for i in range(1, 33)]
    df_iitk = pd.DataFrame()
    
    for sat in PRN:
        file_path_i = f"{basepath}/DPH.IITK.{sat}"
        df = dph_data(file_path_i)
        df_iitk = pd.concat([df_iitk, df], ignore_index=True)
    
    df_iitk = df_iitk[df_iitk['LC cyc'] != 9999]
    df_iitk["LC cyc"] *= 0.19 * 10**3
    df_iitk = df_iitk[df_iitk["Elev"].between(10, 90)]
    df_iitk = df_iitk[df_iitk["LC cyc"].between(-100, 100)]
    
    val1_g = df_iitk["LC cyc"].values
    r1_g = df_iitk['Elev1'].values
    theta1_g = df_iitk['Azimuth'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=600)
    fontsize = 24
    
    ax1.scatter(df_iitk["Elev"], df_iitk["LC cyc"], c='black', s=1, edgecolors='black')
    ax1.set_xlabel('Elevation [deg]', fontsize=fontsize)
    ax1.set_ylabel('LC Phase residual [mm]', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_ylim(-100, 100)
    ax1.set_xlim(0, 90)
    ax1.grid(True)
    
    ax2 = plt.subplot(122, projection='polar')
    sc = ax2.scatter(theta1_g, r1_g, c=val1_g, cmap='seismic', s=10)
    ax2.set_rmax(90)
    ax2.set_yticklabels([])
    ax2.set_rlabel_position(0)
    ax2.grid(True)
    ax2.set_rorigin(0)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_rticks([0, 15, 30, 45, 60, 75, 90])
    
    for i, text in enumerate(["75", "60", "45", "30", "15"]):
        ax2.text(0, (i+1)*15, text, fontsize=fontsize)
    ax2.tick_params(axis='x', which='major', labelsize=fontsize)
    
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.85])
    cbar = fig.colorbar(sc, cax=cbar_ax, ticks=np.arange(-100, 110, 25), extend='both')
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.set_ylabel("LC Phase residual [mm]", fontsize=fontsize)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if output_path:
        plt.savefig(output_path+"dph.png")
        plt.show()
    else:
        plt.show()
    
    
def daily_performance(files_dop,file_obs,file_snr,basepath_gamit,colors, labels,constellations=["C", "E", "R", "G"],save_path =None):
    plot_nsat(files_dop, labels, colors)
    plot_dop(files_dop, labels, colors)
    plot_cnr(file_obs, file_snr, constellations=["C", "E", "R", "G"])
    plot_multipath(file_obs, file_mp)
    plot_lcphase(basepath_gamit)
    return None


#%%
#Example Individual 

#inputs 
files_dop = [
    "D:/c-GNSS/Example/RTKLIB/agri_dop_gps.txt",
    "D:/c-GNSS/Example/RTKLIB/agri_dop_glo.txt",
    "D:/c-GNSS/Example/RTKLIB/agri_dop_gal.txt",
    "D:/c-GNSS/Example/RTKLIB/agri_dop_bds.txt",
    "D:/c-GNSS/Example/RTKLIB/agri_dop.txt",
]

file_obs = "D:/c-GNSS/Example/Observation File/AGRI2880.22O"
file_snr = "D:/c-GNSS/Example/RTKLIB/agri_snr.txt"

labels = ["GPS", "GLONASS", "Galileo", "Beidou", "G+R+E+C"]
colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#377eb8"]

#Call function 
output_path = r"D:/c-GNSS/Example/Output/"  
# make sure to use backword slash at the end of path


#%%Nsat 
plot_nsat(files_dop, labels, colors,output_path)  

#%% DOP 
plot_dop(files_dop, labels, colors,output_path)
#%% CNR

plot_cnr(file_obs, file_snr,output_path)

#%%
plot_multipath(file_obs, file_snr,output_path)

#%%
plot_lcphase(basepath_gamit)




#%%
#Example Combined 




