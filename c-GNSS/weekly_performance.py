# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from matplotlib.ticker import MultipleLocator
import scipy as sp
import seaborn as sns
import matplotlib.gridspec as gridspec
import glob
from datetime import datetime,timedelta




__all__ = ["plot_csmp", "plot_ztd","plot_pos_gamit"]
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
    data = pd.read_csv(file_path, sep='\s+', skiprows=4, names=column_names_handled)
    

    
    
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
    
    '''
    if stat == "mean":
        daily_df = df_ztd1.resample('D').mean()
        daily_df["diff"] = daily_df["diff"]
    elif stat == "std":
        daily_df = df_ztd1.resample('D').std()
        daily_df["diff"] = daily_df["diff"]
    
    
        
    '''
    df_ztd1.reset_index(inplace=True)
    
    #daily_df.reset_index(inplace=True)
        
    #daily_df["DOY"] = daily_df["Epoch_datetime"].dt.dayofyear
    #daily_df = daily_df.iloc[:-1 , :]
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
    gps_epoch = datetime(1980, 1, 6)

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
        fig.savefig(output_path + "weekly_performance_agri.png", bbox_inches='tight', dpi=600)
        plt.show()
    else:
        plt.show()
    plt.close(fig)

