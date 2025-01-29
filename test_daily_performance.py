from c_GNSS.daily_performance import daily_performance

# Example usage
files_dop = [
    "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/FAAP/DOP_SNR/faap_dop_gps.txt",
    "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/FAAP/DOP_SNR/faap_dop_glo.txt",
    "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/FAAP/DOP_SNR/faap_dop_gal.txt",
    "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/FAAP/DOP_SNR/faap_dop_bds.txt",
    "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/FAAP/DOP_SNR/faap_dop.txt",
]

file_obs = "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/AGRI/DOP_SNR/AGRI2880.22O"
file_snr = "E:/OneDrive - IIT Kanpur/Phd Thesis/RP1/Data/LCG/AGRI/DOP_SNR/agri_snr.txt"


labels = ["GPS", "GLONASS", "Galileo", "Beidou", "G+R+E+C"]
colors = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#377eb8"]

daily_performance(files_dop,file_obs,file_snr,colors, labels,constellations=["C", "E", "R", "G"])