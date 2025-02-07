# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def csmp(filepath,output_path=None):
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
    




# Example usage
csmp("E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/results/CSR/AGRI.xlsx")
