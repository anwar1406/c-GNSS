# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:32:49 2025

@author: Dell
"""

import sys
import os
# Get the parent directory of scripts/ (which contains package_folder/)
#sys.path.append(os.path.abspath("../c-GNSS"))
import cGNSS as cg



#%%

import yaml
# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# DAILY
files_dop = config["daily"]["dop_files"]
file_obs = config["daily"]["observation_file"]
file_snr = config["daily"]["snr_file"]
gamit_path = config["daily"]["gamit_path"]

# WEEKLY
file_path_teqc = config["weekly"]["teqc_file"]
filepath_pp = config["weekly"]["positioning_file"]
folder_path_ztd_PPP = config["weekly"]["ztd_ppp_folder"]
folder_path_ztd_GAMIT = config["weekly"]["ztd_gamit_folder"]
basepath_gamit = config["weekly"]["gamit_basepath"]

# OUTPUT
output_path = config["output"]["path"]
#%%
##Daily performance full
cg.daily_performance_plot(file_obs,files_dop,file_snr,gamit_path,output_path)


#%%
##Weekly report Full
cg.weekly_performance_plot(file_path_teqc,filepath_pp,
                           folder_path_ztd_PPP,folder_path_ztd_GAMIT,
                           basepath_gamit,output_path)
