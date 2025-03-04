# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:32:49 2025

@author: Dell
"""

import sys
import os
# Get the parent directory of scripts/ (which contains package_folder/)
sys.path.append(os.path.abspath("../c-GNSS"))
import cGNSS as cg
#%%

##Filepath of processed Data 
#Daily
files_dop = [
    "../Data/AGRI/RTKLIB/agri_dop_gps.txt",
    "../Data/AGRI/RTKLIB/agri_dop_glo.txt",
    "../Data/AGRI/RTKLIB/agri_dop_gal.txt",
    "../Data/AGRI/RTKLIB/agri_dop_bds.txt",
    "../Data/AGRI/RTKLIB/agri_dop.txt",
]

file_obs = "../Data/AGRI/Observation File/AGRI2890.22O"
file_snr = "../Data/AGRI/RTKLIB/agri_snr.txt"
gamit_path = r"../Data/AGRI/GAMIT/289G/" 

#Weekly
file_path_teqc = "../Data/AGRI/Teqc/AGRI.xlsx"
file_path_teqc= "../Data/AGRI/Teqc/AGRI.xlsx"
filepath_pp = "../Data/AGRI/GAMIT/MEAN.AGRI.unk.orbit.res"
folder_path_ztd_PPP = r"../Data/AGRI/ZTD/PPP"
folder_path_ztd_GAMIT =  r"../Data/AGRI/ZTD/GAMIT"
basepath_gamit = "../Data/AGRI/GAMIT/"

#Output path 
output_path = "../Data/AGRI/output/" # make sure to use backword slash at the end of path
#%%
##Daily performance full
cg.daily_performance_plot(file_obs,files_dop,file_snr,gamit_path,output_path)


#%%
##Weekly report Full
cg.weekly_performance_plot(file_path_teqc,filepath_pp,
                           folder_path_ztd_PPP,folder_path_ztd_GAMIT,
                           basepath_gamit,output_path)
