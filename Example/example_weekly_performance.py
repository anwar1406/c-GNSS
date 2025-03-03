# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:32:49 2025

@author: Dell
"""

import sys
import os
# Get the parent directory of scripts/ (which contains package_folder/)
sys.path.append(os.path.abspath("../c-GNSS"))
import weekly_performance as wp
import cGNSS as cg




##Filepath of processed Data 
#Daily
files_dop = [
    "D:/c-GNSS/Example/AGRI/RTKLIB/agri_dop_gps.txt",
    "D:/c-GNSS/Example/AGRI/RTKLIB/agri_dop_glo.txt",
    "D:/c-GNSS/Example/AGRI/RTKLIB/agri_dop_gal.txt",
    "D:/c-GNSS/Example/AGRI/RTKLIB/agri_dop_bds.txt",
    "D:/c-GNSS/Example/AGRI/RTKLIB/agri_dop.txt",
]

file_obs = "D:/c-GNSS/Example/AGRI/Observation File/AGRI2890.22O"
file_snr = "D:/c-GNSS/Example/AGRI/RTKLIB/agri_snr.txt"
gamit_path = r"D:/c-GNSS/Example/AGRI/GAMIT/289G/" 

#Weekly
file_path_teqc = "D:/c-GNSS/Example/AGRI/Teqc/AGRI.xlsx"
file_path_teqc= "D:/c-GNSS/Example/AGRI/Teqc/AGRI.xlsx"
filepath_pp = "D:/c-GNSS/Example/AGRI/GAMIT/MEAN.AGRI.unk.orbit.res"
folder_path_ztd_PPP = r"D:/c-GNSS/Example/AGRI/ZTD/PPP"
folder_path_ztd_GAMIT =  r"D:/c-GNSS/Example/AGRI/ZTD/GAMIT"
basepath_gamit = "D:/c-GNSS/Example/AGRI/GAMIT/"

#Output path 
output_path = "D:/c-GNSS/Example/AGRI/output/"

# make sure to use backword slash at the end of path

##Daily performance full
cg.daily_performance_plot(file_obs,files_dop,file_snr,gamit_path,output_path)

##Weekly report Full
cg.weekly_performance_plot(file_path_teqc,filepath_pp,
                           folder_path_ztd_PPP,folder_path_ztd_GAMIT,
                           basepath_gamit,output_path)
