# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from matplotlib.ticker import MultipleLocator
import scipy as sp


__all__ = ["plot_csmp", "plot_ztd"]
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
    
    # Customize the y-axis and labels
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Multipath [m]", fontsize=40)
    ax2.grid(True, axis="y")
    ax2.set_xlim(xlim_i, xlim_f)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax2.set_xlabel("Day of Year 2022", fontsize=40)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
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
    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=4, names=column_names_handled)
    

    
    
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
    df_ztd["Epoch_datetime"] = df_ztd["Epoch_datetime"].dt.round('S')
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




# Example usage


###Cycle slip
file_path_csmp = "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/results/CSR/AGRI.xlsx"
csmp(file_path_csmp)

###ZTD

path_to_PPP = r"E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/results/ztd_test/PPP"
path_to_gamit = r"E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/results/ztd_test/GAMIT"

plot_ztd(path_to_PPP, path_to_gamit)


