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

#Filepath of processed Data 

file_path_teqc = "D:/c-GNSS/Example/AGRI/Teqc/AGRI.xlsx"
file_path_teqc= "D:/c-GNSS/Example/AGRI/Teqc/AGRI.xlsx"
filepath_pp = "D:/c-GNSS/Example/AGRI/GAMIT/MEAN.AGRI.unk.orbit.res"
folder_path_ztd_PPP = r"D:/c-GNSS/Example/AGRI/ZTD/PPP"
folder_path_ztd_GAMIT =  r"D:/c-GNSS/Example/AGRI/ZTD/GAMIT"
basepath_gamit = "D:/c-GNSS/Example/AGRI/GAMIT/"

#Output path 

output_path = "D:/c-GNSS/Example/AGRI/output/"

##Weekly report Full
wp.weekly_performance_plot(file_path_teqc,filepath_pp,
                           folder_path_ztd_PPP,folder_path_ztd_GAMIT,
                           basepath_gamit,output_path)
