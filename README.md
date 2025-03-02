# c-GNSS
Characterization of a GNSS station

**c-GNSS** is a Python package for daily and weekly performance analysis of GNSS stations. This module uses processed GNSS data from RTKLIB, Teqc, GAMIT/GLOBK and PPP-ARISEN  to generate insightful plots and visualizations, helping analyze the quality of GNSS station.

## Features

### Daily Analysis
1. **NSAT Plot**: Visualize the number of satellites (NSAT) over time for multiple GNSS constellations.
2. **DOP Components Plot**: Visualize GNSS Dilution of Precision (DOP) components:
   - GDOP (Geometric DOP)
   - PDOP (Position DOP)
   - HDOP (Horizontal DOP)
   - VDOP (Vertical DOP)
   - TDOP (Time DOP)
3. **Carrier-to-Noise ratio Plot**: Visualize the carrier-noise ratio as a sky plot.
4. **Multipath Plot**: Visualize the Multipath as a sky plot.
5. **LC Phase residual Plot**: Visualize the LC Phase residual against elevation(Antenna Performance) and Skyplot.

### Weekly Analysis
1. **Cycle slip ratio and Multipath plot**: Visualize the daily cycle slip ratio and moving average multipath over the week.
2. **ZTD Plot**: Visualize the zenith tropospheric delay estimation from PPP-ARISEN and GAMIT to analyse environmental variable performance.
3. **Positioning Plot**: Visualize the post-processed positioning(repeatability) of a station in a baseline processing using GAMIT.
4. **Post-Fit NRMS Plot**: Visualize the GAMIT processed daily post-fit NRMS of the baseline network  as a boxplot.
5. **RMS of one-way Double Difference residuals**: Visualize the GAMIT processed daily RMS of one-way Double Difference residuals of the station of interest as a boxplot.


## Installation

### Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/anwar1406/c-GNSS.git
cd c-GNSS
```


## Contributors
1. Ibaad Anwar
2. Balaji Devaraju
