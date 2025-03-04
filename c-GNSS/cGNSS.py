# -*- coding: utf-8 -*-
"""
Created on Wed Jan  29 15:32:49 2025

@author: Ibaad Anwar
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.ticker import MultipleLocator
import scipy as sp
import seaborn as sns
import matplotlib.gridspec as gridspec
import glob
import datetime
from matplotlib.dates import DateFormatter
from pylab import*

c = 299792458
dtr = np.pi/180

__all__ = ["plot_csmp", "ztd_gamit", "ztd_arisen", "ppp_ztd_hour", "ztd_daily",
           "plot_ztd", "extract_rms_dd", "gamit_pp", "extract_post_fit_nrms",
           "plot_pos_gamit", "extract_station_name",
           "extract_number_of_stations", "weekly_performance_plot", "dop_df",
           "plot_nsat", "plot_dop", "cart2ellipsoid", "antennapos",
           "observationtype", "obsdata_filter", "obsdata_filter_cn",
           "epochdata_filter", "gpstime", "obsdataframe_rtk", "obsdata_snr",
           "plot_cnr", "dph_data", "plot_lcphase", "daily_performance",
           "plot_multipath", "daily_performance_plot"]

def plot_csmp(filepath,output_path=None):
    """
    Plots observation per cyclle slip and multipath values from a .xlsx file generated from Teqc.
    
    Parameters:
    filepath (str): Path to the .xlsx file.
    """
    # Load the data
    df_csmp = pd.read_excel(filepath)
    
    # Create figure and axis
    fig, axes = plt.subplots(figsize=(18, 12), dpi=600)
    
    # Define x-axis limits
    xlim_i = df_csmp["DOY"].iloc[0] - 0.5
    xlim_f = df_csmp["DOY"].iloc[-1] + 0.5
    
    # Plot observation slip
    ax = df_csmp.plot(x='DOY', y=["Obs_slip"], lw=4,
                       fontsize=40, grid=True, color=["#d95f02"], 
                       ax=plt.gca(), legend=False)
    axes.set_ylabel('Obsv/cycle slip', color='#d95f02', fontsize=40)
    axes.set_xlabel("Day of Year 2022", fontsize=40)
    axes.set_ylim(0, 1250)
    axes.set_xlim(xlim_i, xlim_f)
    axes.tick_params(axis='y', labelcolor='#d95f02')
    axes.set_yticks(np.arange(0, 1300, 250))
    
    # Create second y-axis for multipath values
    ax2 = axes.twinx()
    ax2 = df_csmp.plot(x='DOY', y=["mp1", "mp2"], lw=4,
                        fontsize=40, grid=True, color=["#1b9e77", "#7570b3"], 
                        ax=plt.gca())
    colors = ["#1b9e77", "#7570b3"]
    # Customize the y-axis and labels
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Multipath [m]", fontsize=12)
    ax2.grid(True, axis="y")
    ax2.set_xlim(xlim_i, xlim_f)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    #ax2.set_xlabel("Day of Year 2022", fontsize=40)
    ax2.set_yticks(np.arange(0, 1.1, 0.25))
    ax2.legend(["MP1", "MP2"], loc='upper right', fontsize=40)
    
    if output_path:
        plt.savefig(output_path)
        plt.show()
    else:
        plt.show()
    
    # Show plot

def ztd_gamit(file_path):
    """
    Reads data from a file, handling duplicate column names.
    
    Parameters:
    file_path (str): Path to the file to be read.

    Returns:
    pd.DataFrame: DataFrame containing the read data.
    """
    # Function to handle duplicate column names
    def handle_duplicate_columns(names):
        name_count = Counter(names)
        new_names = []
        for name in names:
            if name_count[name] > 1:
                new_name = f"{name}_{name_count[name]}"
                name_count[name] -= 1
                new_names.append(new_name)
            else:
                new_names.append(name)
        return new_names

    # Extracting column names
    with open(file_path, 'r') as file:
        for _ in range(3):
            next(file)  # Skip first 3 lines
        column_names = next(file).split()  # Read the 4th line for column names
        column_names = column_names[1:]
    # Handling duplicate column names
    column_names_handled = handle_duplicate_columns(column_names)

    # Reading the data
    data = pd.read_csv(file_path, sep=r'\s+', skiprows=4, names=column_names_handled)
    

    
    
    data['Date'] = pd.to_datetime(data['Yr'].astype(str) + data['Doy'].astype(str), format='%Y%j')
    data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' +
                                      data['Hr'].astype(str) + ':' +
                                      data['Mn'].astype(str) + ':' +
                                      data['Sec'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')

    data = data[['Datetime','Yr', 'Doy', 'Hr', 'Mn', 'Total']]
    data["ZTD"] = data["Total"]
    data = data[1:]
    return data
def ztd_arisen(file_ztd):
    df_ztd = pd.read_fwf(file_ztd)



    df_ztd = df_ztd[df_ztd["ZTD(m)"]!=0]

    if df_ztd.columns[1] != "Observation time":
        
        df_ztd['Observation time'] = df_ztd.iloc[:, 1] + df_ztd.iloc[:, 2]
        df_ztd['Date'] = df_ztd['Observation time'].str.slice(0,10)
        df_ztd['Hour'] = df_ztd['Observation time'].str.slice(10,12).str.strip()
        df_ztd['Hour'] = df_ztd['Hour'].apply(lambda x: f'0{x}' if int(x) < 10 else str(x))
        df_ztd['Min'] = df_ztd['Observation time'].str.slice(13,15).str.strip()
        df_ztd['Min'] = df_ztd['Min'].apply(lambda x: f'0{x}' if int(x) < 10 else str(x))
        df_ztd['Sec'] = df_ztd['Observation time'].str.slice(16,-1).str.strip()
        df_ztd['Sec'] = df_ztd['Sec'].apply(lambda x: f'0{x}' if float(x) < 10 else str(x))
        df_ztd["datetime"] = df_ztd["Date"] + df_ztd["Hour"] + df_ztd["Min"] + df_ztd["Sec"]
        try:
            df_ztd["Epoch_datetime"] = pd.to_datetime(df_ztd['datetime'],format = '%Y-%m- %d %H%M%S.%f')
        except ValueError:
            df_ztd["Epoch_datetime"] = pd.to_datetime(df_ztd['datetime'],format = '%Y-%m-%d %H%M%S.%f')
            
        
    else:
        df_ztd['Date'] = df_ztd['Observation time'].str.slice(0,10)
        df_ztd['Hour'] = df_ztd['Observation time'].str.slice(11,13).str.strip()
        df_ztd['Hour'] = df_ztd['Hour'].apply(lambda x: f'0{x}' if int(x) < 10 else str(x))
        df_ztd['Min'] = df_ztd['Observation time'].str.slice(14,16).str.strip()
        df_ztd['Min'] = df_ztd['Min'].apply(lambda x: f'0{x}' if int(x) < 10 else str(x))
        df_ztd['Sec'] = df_ztd['Observation time'].str.slice(17,-1).str.strip()
        df_ztd['Sec'] = df_ztd['Sec'].apply(lambda x: f'0{x}' if float(x) < 10 else str(x))
        df_ztd["datetime"] = df_ztd["Date"] + df_ztd["Hour"] + df_ztd["Min"] + df_ztd["Sec"]
        df_ztd["Epoch_datetime"] = pd.to_datetime(df_ztd['datetime'],format = '%Y-%m-%d%H%M%S.%f')
    df_ztd["Epoch_datetime"] = df_ztd["Epoch_datetime"].dt.round('s')
    df_ztd["DOY"] = df_ztd["Epoch_datetime"].dt.dayofyear + df_ztd["Hour"].astype(float)/24 + df_ztd["Min"].astype(float)/(24*60) + df_ztd["Sec"].astype(float)/(24*3600)
    df_ztd = df_ztd[(np.abs(sp.stats.zscore(df_ztd["ZTD(m)"])) < 3)]
    df_ztd = df_ztd[50:]
    return df_ztd
def ppp_ztd_hour(file_path):
    ZTD = []
    Hour = []
    Min = []
    Sec = []
    DOY= []
    df_ztd_f = ztd_arisen(file_path)
    
    doy = df_ztd_f["Epoch_datetime"].iloc[0].timetuple().tm_yday
    year =  df_ztd_f["Epoch_datetime"].iloc[0].year
    
    df_ztd_f["hour"] = df_ztd_f["Hour"].astype(float) + df_ztd_f["Min"].astype(float)/60


    for i in range(0,24):
        result = df_ztd_f["ZTD(m)"][df_ztd_f['hour'].between(i+0.5,i+1.5)]
        
        mean_ztd = 1000*result.mean()
        #if math.isnan(mean_ztd):
            #    print(i)
            #    mean_ztd = df_ztd_f["ZTD(m)"][df_ztd_f['hour'].between(i+0.5,i+1)]
        ZTD.append(mean_ztd)
    
        hour = i+1
    
        if hour ==24:
            Hour.append(str(0))
            DOY.append(doy+1)  
        else:
            Hour.append(str(hour))
            DOY.append(doy) 
        Min.append(str(00))
        Sec.append(str(00))
    
    
    
    df_ztd_hour = pd.DataFrame()
    df_ztd_hour["ZTD"] = ZTD
    df_ztd_hour["Year"] = year
    df_ztd_hour["DOY"] = DOY
    df_ztd_hour["Hour"] = Hour
    df_ztd_hour["Mn"] = Min
    df_ztd_hour["Sec"] = Sec
    df_ztd_hour['Date'] = pd.to_datetime(df_ztd_hour['Year'].astype(str) + df_ztd_hour['DOY'].astype(str), format='%Y%j')
    df_ztd_hour['Datetime'] = pd.to_datetime(df_ztd_hour['Date'].astype(str) + ' ' +
                                             df_ztd_hour['Hour'].astype(str) + ':' +
                                             df_ztd_hour['Mn'].astype(str) + ':' +
                                             df_ztd_hour['Sec'].astype(str), format='%Y-%m-%d %H:%M:%S')

    
    df_ztd_hour["DOY_frac"] = df_ztd_hour["DOY"].astype(float) + df_ztd_hour["Hour"].astype(float)/24
    return df_ztd_hour



def ztd_daily(path_to_PPP,path_to_gamit):
    
    def list_folders(path_to_PPP):
        try:
            # List all items in the directory
            items = os.listdir(path_to_PPP)
            
            # Filter out only directories
            folders = sorted([item for item in items if os.path.isdir(os.path.join(path_to_PPP, item))])
            
            return folders
        except Exception as e:
            print(f"Error: {e}")
            return []
        
    folders = list_folders(path_to_PPP)
    
    station_name = folders[0][0:4]
    year = folders[0][4:8]
    doy_i = int(folders[0][8:11])
    doy_f = int(folders[-1][8:11])
    ac  = folders[0].split("_")[1]
    strag  = folders[0].split("_")[3]
    ppp_model = folders[0].split("_")[4]
    constell = folders[0].split("_")[-1]
    
    
    date = np.arange(doy_i,doy_f)
    df_ZTD_f_gm = pd.DataFrame()
    df_ZTD_f_pp = pd.DataFrame()
    for i in date:
        if len(str(i))<3:
            i = "0"+str(i)
        else:
            i = str(i)
            
            
        
        file_pp_faap = path_to_PPP+"/"+station_name+year+i+"_"+ac+"_Kalman_"+strag+"_"+ ppp_model+"_"+constell+"/"+station_name+year+i+"_ZTD_Clock.txt"
        
        df_ztd_hour = ppp_ztd_hour(file_pp_faap)
        
        df_ZTD_f_pp = pd.concat([df_ZTD_f_pp,df_ztd_hour])
        
        
        
        
        
        file_gm_faap = path_to_gamit + "/met_"+station_name.lower()+"."+year[2:]+i
        ztd_gamit_faap = ztd_gamit(file_gm_faap)
        
        
        df_ZTD_f_gm = pd.concat([df_ZTD_f_gm,ztd_gamit_faap])
    df_ZTD_f_gm.reset_index(inplace = True)
    df_ZTD_f_pp.reset_index(inplace = True)
        
        
    df_ZTD = pd.merge(df_ZTD_f_gm, df_ZTD_f_pp,  how='left',on=['Datetime'])
    df_ztd1 = pd.DataFrame()
    df_ztd1["Epoch_datetime"] = df_ZTD["Datetime"]
    df_ztd1["DOY_frac"] = df_ZTD["DOY_frac"]
    df_ztd1["GAMIT"] = df_ZTD["ZTD_x"]
    df_ztd1["PPP"] = df_ZTD["ZTD_y"]
    df_ztd1["diff"] = df_ZTD["ZTD_y"] - df_ZTD["ZTD_x"]
    df_ztd1.set_index('Epoch_datetime', inplace=True)
    
    df_ztd1.reset_index(inplace=True)
    
    return df_ztd1

def plot_ztd(path_to_PPP, path_to_gamit,output_path=None):
    """
    Plots Zenith Tropospheric Delay (ZTD) from PPP and GAMIT results.

    Parameters:
    path_to_PPP (str): Path to PPP results directory.
    path_to_gamit (str): Path to GAMIT results directory.

    Returns:
    None (Displays the plot)
    """
    # Process data
    df_ztd = ztd_daily(path_to_PPP, path_to_gamit)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(18, 10), dpi=600)

    # Plot data
    df_ztd.plot(x='DOY_frac', y=["GAMIT", "PPP"], lw=2,
                fontsize=40, grid=True, color=["#1b9e77", "#d95f02"], ax=ax)

    # Labels and formatting
    ax.set_ylabel("ZTD [mm]", fontsize=40)
    ax.set_xlabel(f"Day of year {df_ztd['Epoch_datetime'].iloc[0].year}", fontsize=40)

    # Set y-axis limits rounded to the nearest 50
    min_ztd = np.round((df_ztd[["GAMIT", "PPP"]].min().min() - 50) / 50) * 50
    max_ztd = np.round((df_ztd[["GAMIT", "PPP"]].max().max() + 50) / 50) * 50
    ax.set_ylim(min_ztd, max_ztd)

    # Set major y-axis ticks at intervals of 50
    ax.yaxis.set_major_locator(MultipleLocator(50))

    # Customize tick parameters
    ax.tick_params(axis='x', labelrotation=0)

    # Set legend
    ax.legend(labels=["GAMIT", "PPP-ARISEN"], loc="upper right", fontsize=40, ncol=1)

    # Fill the area between the two curves
    ax.fill_between(df_ztd['DOY_frac'], df_ztd['GAMIT'], df_ztd['PPP'], color='#377eb8', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.show()
    else:
        plt.show()
        
def extract_rms_dd(file_path,doy):
    """
    Extracts data from a text file between two specified lines.

    Parameters:
    file_path (str): Path to the text file.
    start_marker (str): The line indicating the start of data extraction.
    end_marker (str): The line indicating the end of data extraction.

    Returns:
    pd.DataFrame: DataFrame containing the extracted data.
    """
    # Initialize variables to store the extracted data
    
    start_marker = "RMS by site and satellite (mm)"
    end_marker = "Number of data by site and satellite"
    
    data_between_lines = []
    start_collecting = False

    # Read the file and process the lines
    with open(file_path, 'r') as file:
        for line in file:
            if start_marker in line:
                start_collecting = True
            elif end_marker in line:
                start_collecting = False
            if start_collecting:
                data_between_lines.append(line.strip().split())

    # Remove the first line since it's the starting marker
    if data_between_lines:
        data_between_lines.pop(0)
        data_between_lines.pop(-1)

    df = pd.DataFrame(data_between_lines)
        
    df_req = df.iloc[1:-1,2:4]
    df_req = df_req.transpose()
    df_req.columns = df_req.iloc[0]
    df_req = df_req.drop(df_req.index[0])
    df_req["DOY"] = doy
    return df_req


def gamit_pp(file_path):
    import pandas as pd
    import datetime

    # Function to parse date and extract day of year
    def extract_day_of_year(date_str):
        try:
            date = datetime.datetime.strptime(date_str, '%Y%m%d')
            return date.timetuple().tm_yday
        except ValueError:
            return None

    # Function to extract numeric data and DOY from a line based on the header format
    def extract_data_and_doy(line, header_parts):
        parts = line.split()
        data = {}
        for column in ['N', 'E', 'U','dN','ne', 'dE','ee','dU','ue']:
            if column in header_parts:
                index = header_parts.index(column)
                if index < len(parts):
                    try:
                        data[column] = float(parts[index])
                    except ValueError:
                        data[column] = None

        # Extracting Day of Year from the YYYYMMDD part
        date_str = parts[0]  # Assuming the date is the first part of the line
        data['DayOfYear'] = extract_day_of_year(date_str)
        data['year']  = parts[0][0:4]
        return data

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # The header line gives us an indication of the format
    header_line = 'YYYYMMDD HHMNSC    DecYr     MJD              N         E         U        dN       ne    F       dE       ee    F        dU       ue   F'
    header_parts = header_line.split()

    # Extracting data including Day of Year
    extracted_data_with_doy = []
    for line in lines:
        if len(line.split()) == len(header_parts):
            data_with_doy = extract_data_and_doy(line, header_parts)
            extracted_data_with_doy.append(data_with_doy)

    # Creating a DataFrame from the extracted data
    
    df = pd.DataFrame(extracted_data_with_doy)
    return df
def extract_post_fit_nrms(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Find the line containing "Double difference statistics" and extract the post-fit NRMS value from the next line
    dd_statistics_index = None
    for i, line in enumerate(content):
        if "Double difference statistics" in line:
            dd_statistics_index = i
            break

    # Extract the next line which should contain the post-fit NRMS value
    if dd_statistics_index is not None and dd_statistics_index + 1 < len(content):
        post_fit_nrms_line1 = content[dd_statistics_index + 1]
        post_fit_nrms_line2 = content[dd_statistics_index + 2]
        post_fit_nrms_line3 = content[dd_statistics_index + 3]
        post_fit_nrms_line4 = content[dd_statistics_index + 4]
        
        # Extract the post-fit NRMS value from the line
        post_fit_nrms1 = post_fit_nrms_line1.split("Postfit nrms:")[1].strip()
        post_fit_nrms2 = post_fit_nrms_line2.split("Postfit nrms:")[1].strip()
        post_fit_nrms3 = post_fit_nrms_line3.split("Postfit nrms:")[1].strip()
        post_fit_nrms4 = post_fit_nrms_line4.split("Postfit nrms:")[1].strip()
              
        return float(post_fit_nrms1),float(post_fit_nrms2),float(post_fit_nrms3),float(post_fit_nrms4)
    else:
        return None
    

def plot_pos_gamit(filepath_pp,output_path=None):
    df_pp = gamit_pp(filepath_pp)
    fig, ax = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(16, 14), 
                           gridspec_kw={'width_ratios': [5, 1]}, dpi=600)
    
    # Error bar plots
    ax[0][0].errorbar(df_pp["DayOfYear"], df_pp["dN"], yerr=df_pp["ne"], fmt='o', capsize=8, 
                      elinewidth=3, markersize=8, color='black', ecolor='grey')
    ax[1][0].errorbar(df_pp["DayOfYear"], df_pp["dE"], yerr=df_pp["ee"], fmt='o', capsize=8, 
                      elinewidth=3, markersize=8, color='black', ecolor='grey')
    ax[2][0].errorbar(df_pp["DayOfYear"], df_pp["dU"], yerr=df_pp["ue"], fmt='o', capsize=8, 
                      elinewidth=3, markersize=8, color='black', ecolor='grey')
    
    # Y-axis limits and ticks
    y_limits = [(-3, 3), (-3, 3), (-15, 15)]
    y_ticks = [np.arange(-3, 4, 1), np.arange(-3, 4, 1), np.arange(-15, 20, 5)]
    for i in range(3):
        ax[i][0].set_ylim(y_limits[i])
        ax[i][0].set_yticks(y_ticks[i])
        ax[i][0].grid(True)
        ax[i][0].tick_params(axis='both', which='major', labelsize=30)
    
    ax[2][0].tick_params(axis='y', which='major', labelsize=30, colors="#1b9e77")
    ax[2][0].tick_params(axis='x', which='major', labelsize=30)
    
    # Titles and labels
    ax[0][0].set_title("Positioning", fontsize=30)
    ax[2][0].set_xlabel("Day of year {}".format(df_pp["year"][0]), fontsize=30)
    ax[0][0].set_ylabel("North [mm]", fontsize=30)
    ax[1][0].set_ylabel("East [mm]", fontsize=30)
    ax[2][0].set_ylabel("Up [mm]", fontsize=30, color="#1b9e77")
    
    # X-axis settings
    ax[2][0].set_xticks(np.arange(df_pp["DayOfYear"].iloc[0], df_pp["DayOfYear"].iloc[-1] + 1, 1))
    ax[2][0].set_xlim(df_pp["DayOfYear"].iloc[0] - 0.6, df_pp["DayOfYear"].iloc[-1] + 1)
    
    # Violin plots
    for i in range(3):
        ax[i][1].grid(axis="y")
    
    sns.violinplot(y=df_pp["dN"], ax=ax[0][1], inner="box", linewidth=2.5, fill=False)
    sns.violinplot(y=df_pp["dE"], ax=ax[1][1], inner="box", linewidth=2.5, fill=False)
    sns.violinplot(y=df_pp["dU"], ax=ax[2][1], inner="box", linewidth=2.5, fill=False)
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path+"positioning.png")
        plt.show()
    else:
        plt.show()


def extract_station_name(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "Station name" in line:
                return line.split(":")[1].split("_")[0].strip()
    return None


# Define a function to extract the number of stations used from the file
def extract_number_of_stations(file_path):
    with open(file_path, "r") as file:
        for line in file:
            if "Number of stations used" in line:
                return int(line.split()[4])  # Extract the number from the string

def weekly_performance_plot(file_path_teqc,filepath_pp,folder_path_ztd_PPP,folder_path_ztd_GAMIT,basepath_gamit, output_path=None):
    
    station_name = extract_station_name(filepath_pp)
    ##Cycle slip and multipath from teqc 
    df_csmp = pd.read_excel(file_path_teqc)
    df_csmp["DOY"] = df_csmp["DOY"] - df_csmp["DOY"][0]
    
    #GAMIT solution
    df_pp = gamit_pp(filepath_pp)
    
    df_ztd = ztd_daily(folder_path_ztd_PPP, folder_path_ztd_GAMIT)
    
    
    fontsize = 14
    a4_width, a4_height = 8.27, 11.69
    fig = plt.figure(figsize=(a4_width, a4_height), dpi=600)
    gs = gridspec.GridSpec(8, 3, height_ratios=[1.5, 2, 0.8, 0.9, 0.9, 0.5, 1, 1], hspace=0.2)

    xlim_i = df_csmp["DOY"].iloc[0] - 0.5
    xlim_f = df_csmp["DOY"].iloc[-1] + 0.5

    # First subplot (Obs Slip)
    ax1 = fig.add_subplot(gs[0, :])
    df_csmp.plot(x='DOY', y=["Obs_slip"], lw=2, fontsize=fontsize, grid=True, color=["#d95f02"], ax=ax1, legend=False)
    ax1.set_ylabel('Obs/CS', color='#d95f02', fontsize=fontsize)
    ax1.set_ylim(0, 1200)
    ax1.set_xlim(xlim_i, xlim_f)
    ax1.set_yticks(np.arange(0, 1300, 300))
    ax1.tick_params(axis='y', labelcolor='#d95f02')
    ax1.spines['bottom'].set_visible(False)
    
    ax2 = ax1.twinx()
    df_csmp.plot(x='DOY', y=["mp1", "mp2"], lw=2, fontsize=fontsize, grid=True, color=["#1b9e77", "#7570b3"], ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Multipath [m]", fontsize=fontsize)
    ax2.set_xlim(xlim_i, xlim_f)
    ax2.set_yticks(np.arange(0, 1.1, 0.25))
    ax2.legend(["MP1", "MP2"], labelcolor=["#1b9e77", "#7570b3"], handletextpad=0, handlelength=0, loc='upper right', ncol=2, fontsize=fontsize+2)
    
    
    # Second subplot (ZTD)
    ax3 = fig.add_subplot(gs[1, :], sharex=ax1)
    df_ztd['DOY_frac'] -= df_ztd['DOY_frac'][0]
    df_ztd.plot(x='DOY_frac', y=["GAMIT", "PPP"], lw=2, fontsize=fontsize, grid=True, color=["#1b9e77", "#d95f02"], ax=ax3)
    ax3.set_ylabel("ZTD [mm]", fontsize=fontsize)
    
    min_ztd = np.round((df_ztd[["GAMIT", "PPP"]].min().min() - 50) / 50) * 50
    max_ztd = np.round((df_ztd[["GAMIT", "PPP"]].max().max() + 50) / 50) * 50
    ax3.set_ylim(min_ztd, max_ztd)
    ax3.yaxis.set_major_locator(MultipleLocator(50))
    ax3.legend(labels=["GAMIT", "PPP-ARISEN"], labelcolor=["#1b9e77", "#d95f02"], handletextpad=0, handlelength=0, loc="upper right", fontsize=fontsize+2, ncol=2)
    ax3.fill_between(df_ztd['DOY_frac'], df_ztd['GAMIT'], df_ztd['PPP'], color='#377eb8', alpha=0.3)
    ax3.tick_params(labelbottom=False)
    ax3.set_xlabel(" ")
    
    ##GPS Week 
    
    # GPS epoch start date
    gps_epoch = datetime.datetime(1980, 1, 6)

    # Compute GPS Week
    gps_week  = ((df_ztd["Epoch_datetime"][0] - gps_epoch).days // 7)

    df_pp["DayOfYear"] -= df_pp["DayOfYear"].iloc[0]
    
    for i, (var, err, label, y_lim, y_ticks) in enumerate(
        zip(["dN", "dE", "dU"], ["ne", "ee", "ue"], ["N [mm]", "E [mm]", "U [mm]"],
            [(-5, 5), (-5, 5), (-15, 15)], [np.arange(-5, 6, 5), np.arange(-5, 6, 5), np.arange(-15, 20, 15)])):

        ax = fig.add_subplot(gs[i + 2, :], sharex=ax1)
        ax.errorbar(df_pp["DayOfYear"], df_pp[var], yerr=df_pp[err], fmt='o', capsize=4, elinewidth=1, markersize=4, color='black', ecolor='grey')
        ax.set_ylim(y_lim)
        ax.set_yticks(y_ticks)
        ax.set_ylabel(label, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True)
        if i < 2:
            ax.tick_params(labelbottom=False)
            ax.set_ylabel(label, labelpad=18)
        else:
            ax.set_ylabel("U [mm]", fontsize=fontsize, color="#1b9e77",labelpad = 8)
            ax.tick_params(axis='y', which='major', labelsize=fontsize, colors="#1b9e77")
            ax.set_xlabel("Day of GPS week {}".format(gps_week), fontsize=fontsize)
    
    ax6 = fig.add_subplot(gs[6:, :2])
    file_list = sorted(glob.glob(basepath_gamit + "/*.summary"))
    f_i = int(file_list[0].split("_")[2][:3])
    
    f_f = int(file_list[-1].split("_")[2][:3])
    num_stations = extract_number_of_stations(file_list[0])
    
    
    df_nrms = pd.DataFrame([extract_post_fit_nrms(file) for file in file_list], columns=['Constrained Free', 'Constrained Fixed', 'Loose Free', 'Loose Fixed'])
    boxplots2  = ax6.boxplot([df_nrms[col] for col in df_nrms.columns], labels=df_nrms.columns, flierprops=dict(marker=".", markerfacecolor="black", markersize=12))
    for median in boxplots2['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    for i, column in enumerate(df_nrms.columns):
        y_max = df_nrms[column].max()  # Get max value for positioning
        ax6.text(i + 1, y_max + 0.0005,column.replace(" ", "\n"), ha='center', fontsize=12)
    ax6.set_xticks([])
    ax6.tick_params(axis='y', which='major', labelsize=fontsize)
    ax6.set_title('Post-fit NRMS', fontsize=fontsize)
    ax6.set_ylim(0.19, 0.22)
    ax6.grid(True, axis="y")
    
    ax7 = fig.add_subplot(gs[6:, 2])
    
    
    
    df2 = pd.concat([extract_rms_dd(basepath_gamit + f"{i}G/autcln.post.sum", i) for i in range(f_i, f_f+1)], axis=0)
    
    
    df2 = df2[[station_name]].replace('nan', np.nan).apply(pd.to_numeric)
    boxplot = ax7.boxplot([df2[col] for col in df2.columns], labels=df2.columns, flierprops=dict(marker=".", markerfacecolor="black", markersize=12))
    for median in boxplot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax7.set_title('RMS of DD \n Post-Fit Res [mm]', fontsize=fontsize)
    ax7.set_ylim(4, 12)
    ax7.grid(axis="y")
    ax7.set_xticklabels([])
    ax7.tick_params(axis='both', which='major', labelsize=fontsize)
    ax7.text(0.925, -1.5, num_stations,
             transform=ax.transAxes,  # Use axis coordinates (0 to 1)
             fontsize=fontsize,
             verticalalignment='bottom',
             #horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path +station_name.lower()+ "weekly_performance.png", bbox_inches='tight', dpi=600)
        plt.show()
    else:
        plt.show()
    plt.close(fig)

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
    df["TDOP"] = np.where(df["GDOP"] ** 2 >= df["PDOP"] ** 2, 
                      np.sqrt(df["GDOP"] ** 2 - df["PDOP"] ** 2), 
                      np.nan)
    df["Epoch_datetime"] = pd.to_datetime(df['Epoch'], format='%Y/%m/%d %H:%M:%S.%f')
    return df
def plot_nsat(file_paths, labels, colors, gs_inner, fig):
    """
    Plot NSAT over time using a nested GridSpec layout inside an existing figure.

    Args:
        file_paths (list of str): Paths to the input DOP data files.
        labels (list of str): Labels for each dataset to appear in the legend.
        colors (list of str): Colors for each dataset's plot line.
        gs_inner (gridspec.GridSpecFromSubplotSpec): Nested GridSpec for subplot allocation.
        fig (matplotlib.figure.Figure): The main figure object.
    """
    ax1 = fig.add_subplot(gs_inner[0, 0])
    ax2 = fig.add_subplot(gs_inner[1, 0], sharex=ax1)

    date_form = DateFormatter("%H")
    linecolor =colors
    for file_path, label, color in zip(file_paths, labels, colors):
        df = dop_df(file_path)
        date_str = df["Epoch"].iloc[15].split(" ")[0]
        date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d")
        formatted_date = date_obj.strftime("UTC %d %b %Y")

        if label == "G+R+E+C":
            ax1.plot(df["Epoch_datetime"], df["NSAT"], color=color, label=label, lw=1.5)
        else:
            ax2.plot(df["Epoch_datetime"], df["NSAT"], color=color, label=label, lw=1.5)
        
    # Top subplot (ax1) settings
    ax1.xaxis.set_major_formatter(date_form)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_yticks(np.arange(24, 57, 8))
    ax1.set_ylim(16, 56)
    ax1.set_xlabel(" ")
    ax1.set_xticklabels([])
    ax1.set_ylabel("NSAT", fontsize=12)
    ax1.legend(ncol=1, loc="upper right", fontsize=14,labelcolor = linecolor[-1],handletextpad=0,handlelength=0)
    
    
    ax1.set_xticklabels([])
    ax1.tick_params(labelbottom=False)
    
    # Bottom subplot (ax2) settings
    ax2.xaxis.set_major_formatter(date_form)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_yticks(np.arange(0, 17, 4))
    ax2.set_xlabel(formatted_date, fontsize=12)
    ax2.set_ylabel("NSAT", fontsize=12)
    ax2.axhline(y=4, color="black", linestyle="--", linewidth=2)
    ax2.legend(ncol=4, loc="upper right", fontsize=14,labelcolor = linecolor,handletextpad=0,handlelength=0)
    ax2.text(ax2.get_xticks()[2], 3, "NSAT threshold", color="black", fontsize=12, verticalalignment="top")
    return ax1, ax2
def plot_dop(file_paths, labels, colors, gs_inner, fig):
    """
    Plot DOP components (GDOP, PDOP, HDOP, VDOP, TDOP) for multiple GNSS systems.

    Args:
        file_paths (list of str): Paths to the input DOP data files.
        labels (list of str): Labels for the GNSS systems.
        colors (list of str): Colors for DOP components.
        gs_inner (gridspec.GridSpecFromSubplotSpec): Nested GridSpec for subplot allocation.
        fig (matplotlib.figure.Figure): The main figure object.
    """
    date_form = DateFormatter("%H")

    # Fix: Initialize axes properly before using sharex
    axes = []
    for i in range(5):
        if i == 0:
            ax = fig.add_subplot(gs_inner[i, 0])
        else:
            ax = fig.add_subplot(gs_inner[i, 0], sharex=axes[0])  # Share x-axis with first subplot
        axes.append(ax)

    dop_data = [dop_df(path) for path in file_paths]
    
    for i, (df, label) in enumerate(zip(dop_data[::-1], labels[::-1])):
        axes[i].plot(df["Epoch_datetime"], df["GDOP"], color=colors[0], label="GDOP")
        axes[i].plot(df["Epoch_datetime"], df["PDOP"], color=colors[1], label="PDOP")
        axes[i].plot(df["Epoch_datetime"], df["HDOP"], color=colors[2], label="HDOP")
        axes[i].plot(df["Epoch_datetime"], df["VDOP"], color=colors[3], label="VDOP")
        axes[i].plot(df["Epoch_datetime"], df["TDOP"], color=colors[4], label="TDOP")

        axes[i].xaxis.set_major_formatter(date_form)
        axes[i].grid(True)
        axes[i].set_ylim([0, 6])
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].set_yticks(np.arange(0, 7, 2))
        axes[i].set_ylabel(label, fontsize=12)
        linecolor = colors
        if i == 0:
            axes[i].legend(labelcolor = linecolor,handletextpad=0,handlelength=0,ncol=5, bbox_to_anchor=(1, 1.1), loc="upper right", fontsize=14)
        if i < 4:
            axes[i].set_xticklabels([])
            axes[i].tick_params(labelbottom=False)
            
        if i < 4:
            axes[i].set_yticklabels([])
            axes[i].tick_params(labelbottom=False)
    # X-axis label only for the last subplot
    date_str = df["Epoch"].iloc[15].split(" ")[0]
    date_obj = datetime.datetime.strptime(date_str, "%Y/%m/%d")
    formatted_date = date_obj.strftime("UTC %d %b %Y")
    axes[-1].set_xlabel(formatted_date, fontsize=12)
    
          # Remove tick labels
            #axes[i].xaxis.set_visible(False)  # Hide x-axis completely


    return axes



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

def plot_cnr(file_obs, file_snr, gs_inner, fig, constellations=["G", "R", "E", "C"], cmap="plasma", snr_min=10, snr_max=55):
    """
    Generate horizontal polar scatter plots of CNR values for specified constellations.

    Parameters:
    -----------
    file_obs : str
        File path of the observation data (RINEX file).
    file_snr : str
        File path of the SNR data (from RTKLIB or similar).
    gs_inner : gridspec.GridSpecFromSubplotSpec
        Nested GridSpec for subplot allocation.
    fig : matplotlib.figure.Figure
        The main figure object.
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
    df_obs["Elev_Plot"] = 90 - df_obs.Elevation           # Convert elevation to polar plot scale

    # Initialize axes list
    axes = []

    for idx, constellation in enumerate(constellations):
        ax = fig.add_subplot(gs_inner[0, idx], projection="polar")  # Create horizontal subplots
        axes.append(ax)

        df_constellation = df_obs[df_obs["constellation"] == constellation]
        if df_constellation.empty:
            continue

        val = df_constellation["SNR"].values
        r = df_constellation['Elev_Plot'].values
        theta = df_constellation['Azimuth_r'].values

        scatter = ax.scatter(theta, r, c=val, cmap=cmap, s=1.5, vmin=snr_min, vmax=snr_max)
        ax.set_rmax(90)
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_rorigin(0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_title(constellation, fontsize=14,y = 0.85,color="black", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        
        if constellation != "R":
            ax.set_xticklabels([])  # Remove tick labels
            ax.tick_params(labelbottom=False)
        if constellation == "G":  
            ax.text(0, 1.3 * ax.get_rmax(), r"L1 $\mathrm{C/N_0\ [dBHz]}$", fontsize=14, ha="center",bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if constellation == "C":
            ax.text(0,-15,'75')
            ax.text(0,-30,'60')
            ax.text(0,-45,'45')
            ax.text(0,-60,'30')
            ax.text(0,-75,'15')
        
    # Add a horizontal colorbar below the plots
    cbar_ax = fig.add_axes([0.02, 0.39,0.015, 0.19])  # left bottom width height 
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical', ticks=np.arange(snr_min, snr_max + 5, 5))
    cbar.ax.tick_params(labelsize=12)
    #cbar.ax.set_ylabel("L1 " + r"$\mathrm{C/N_0(db-Hz)}$", fontsize=12, labelpad=5)

    return axes
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

def plot_lcphase(basepath,station_name, ax, fig):
    """
    Function to plot LC Phase residuals inside a given axis.

    Parameters:
    -----------
    basepath : str
        Base path where DPH.AGRI.PRNi files are located.
    ax : matplotlib axis
        Axis to plot on.
    fig : matplotlib.figure.Figure
        The main figure object.
    fontsize : int, optional
        Font size for labels (default is 12).
    """
    fontsize=12
    PRN = [f'PRN{i:02d}' for i in range(1, 33)]
    df_iitk = pd.DataFrame()
    
    for sat in PRN:
        file_path_i = f"{basepath}/DPH."+station_name+f".{sat}"
        df = dph_data(file_path_i)
        df_iitk = pd.concat([df_iitk, df], ignore_index=True)
    
    df_iitk = df_iitk[df_iitk['LC cyc'] != 9999]
    df_iitk["LC cyc"] *= 0.19 * 10**3
    df_iitk = df_iitk[df_iitk["Elev"].between(10, 90)]
    df_iitk = df_iitk[df_iitk["LC cyc"].between(-100, 100)]

    # Scatter plot
    ax.scatter(df_iitk["Elev"], df_iitk["LC cyc"], c='black', s=1, edgecolors='black')
    ax.set_xlabel('Elevation [deg]', fontsize=fontsize)
    #ax.set_ylabel('LC Phase residual [mm]', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylim(-100, 100)
    ax.set_xlim(0, 90)
    ax.grid(True)
    ax.set_ylabel(None)
    # Adjust subplot position to remove left padding
    box = ax.get_position()
    ax.set_position([box.x0 - 0.05, box.y0, box.width * 1.1, box.height])  # Shift left and expand
    ax.set_title(
        "LC Phase Residuals [mm]", 
        fontsize=fontsize, 
        color="black", 
        pad=15,  # Space between title and plot
        y=1.05,  # Moves title slightly up
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )

def daily_performance(files_dop,file_obs,file_snr,basepath_gamit,colors, labels,outpath_path=None):
    plot_nsat(files_dop, labels, colors,output_path) 
    plot_dop(files_dop, labels, colors,output_path)
    plot_cnr(file_obs, file_snr,output_path)
    plot_multipath(file_obs, file_mp)
    plot_lcphase(basepath_gamit)
    return None

def plot_multipath(file_obs, file_snr, ax, fig):
    """
    Function to plot L1 Pseudorange Multipath as a polar plot inside a given axis.
    
    Parameters:
    -----------
    file_obs : str
        File path of the observation data (RINEX file).
    file_snr : str
        File path of the SNR data (from RTKLIB or similar).
    ax : matplotlib axis
        Axis to plot on.
    fig : matplotlib.figure.Figure
        The main figure object.
    fontsize : int, optional
        Font size for labels (default is 24).
    vmax : float, optional
        Maximum colorbar limit (default is 1).
    vmin : float, optional
        Minimum colorbar limit (default is -1).
    """
    fontsize=12
    vmax=1
    vmin=-1
    df_obs2 = obsdata_snr(file_obs, file_snr)
    df_obs2_g = df_obs2[df_obs2["constellation"] == "G"]
    df_look = df_obs2_g[["Epoch_datetime", 'Elevation', 'Azimuth_r', 'Elev_Plot', "MP"]]

    cmap = cm.get_cmap('RdGy', 8)

    val = df_look["MP"].values
    r = df_look['Elev_Plot'].values
    theta = df_look['Azimuth_r'].values

    sc = ax.scatter(theta, r, c=val, cmap=cmap, s=10, vmin=vmin, vmax=vmax)
    ax.set_rmax(90)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rlabel_position(0)
    ax.grid(True)
    ax.set_rorigin(0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])

    # Add text labels for elevation rings
    for i, text in enumerate(["75", "60", "45", "30", "15"]):
        ax.text(0, (i+1)*15, text, fontsize=fontsize, ha='center', va='center')

    ax.tick_params(axis='x', which='major', labelsize=fontsize)

    # Move colorbar to the left side
    cbar_ax = fig.add_axes([0.02, 0.038, 0.015, 0.275])  # Left, Bottom, Width, Height
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical', ticks=np.arange(vmin, vmax + 0.1, 0.25))
    cbar.ax.tick_params(labelsize=fontsize)
    #cbar.ax.set_ylabel("L1 Pseudorange Multipath [m]", fontsize=fontsize, labelpad=10)

    # Adjust subplot position to remove left padding
    box = ax.get_position()
    ax.set_position([box.x0 - 0.05, box.y0, box.width * 1.1, box.height])  # Shift left and expand
    ax.text(0, 1.2 * ax.get_rmax(), "GPS L1 Pseudorange \n Multipath [m]", fontsize=12, ha="center",bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


def daily_performance_plot(file_obs,files_dop,file_snr,gamit_path,output_path = None):
    # A4 size in inches
    station_name = file_obs.split("/")[-1][:4]
    labels = ["G", "R", "E", "C", "G+R+E+C"]
    colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#377eb8"]
    
    a4_width, a4_height = 8.27, 11.69  
    fig = plt.figure(figsize=(a4_width, a4_height), dpi=600)

    # Define GridSpec layout with custom row heights
    gs = gridspec.GridSpec(3, 6, height_ratios=[1.5, 1, 1.1])  # Adjusted height ratios
    
    # Allocate `2,1` space inside ax1 for plot_nsat
    gs_inner_nsat = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0:2], height_ratios=[1, 2], hspace=0)

    # Allocate `5,1` space inside ax2 for plot_dop
    gs_inner_dop = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, 2:], hspace=0.05)

    # Allocate `4,1` space inside ax3 for plot_cnr
    gs_inner_cnr = gridspec.GridSpecFromSubplotSpec(1, len(["G", "R", "E", "C"]), subplot_spec=gs[1, :], wspace=0.05,hspace = 0)

    # Pass nested GridSpec to plot_nsat
    plot_nsat(files_dop, labels, colors, gs_inner_nsat, fig)

    # Pass nested GridSpec to plot_dop
    plot_dop(files_dop, labels, colors, gs_inner_dop, fig)

    # Pass nested GridSpec to plot_cnr
    plot_cnr(file_obs, file_snr, gs_inner_cnr, fig)

    # Allocate `ax4` for plot_multipath and `ax5` for plot_lcphase
    ax4 = fig.add_subplot(gs[2, 0:3], projection="polar")  # Spanning first 3 columns
    ax5 = fig.add_subplot(gs[2, 3:])  # Remaining columns for lcphase

    # Plot Multipath Data in ax4
    plot_multipath(file_obs, file_snr, ax4, fig)

    # Plot LC Phase Data in ax5
    plot_lcphase(gamit_path,station_name,ax5, fig)
    plt.tight_layout()
    plt.show()
    
    if output_path:
        fig.savefig(output_path+station_name.lower()+"_daily_perfromance.png",dpi = 600)
        plt.show()
    else:
        plt.show()













