# ArditArifi

# Purpose
This project investigates the impact of natural aerosol emissions on the polar regions by comparing different models from the CMIP6 experiment. Specifically, it examines how these emissions influence the radiation budget and contribute to polar warming. Currently, only the 'rtmt' variable (net downard radiation flux on TOA) from CMIP6 is analyzed.

# Description
(rtmt.py)
The main script in this project. It takes the CMIP6 data, evaluates mean and outputs the multi model mean of rtmt between historical aerosol emission (control) and aersol emissions upon aerosol perturbation (experiments) as a function of season + region ( ARC = arctic, ANT = antarctica )

Configuration:
CMIP6 data is located - source_path
output should be stored - output_path
experiments and models can be also modified as for preferences. 

A file ensemble_mean_barplot.png will be created.

(area_weight.py)
This script compares different area-weight functions used in atmospheric physics for averaging. It demonstrates that, while different functions yield negligible differences at standard model grid resolutions, area-weighted averaging is generally necessary. The script considers only regular grids.

Configuration:

Wished grid resolution in degress:
lat_res = 0.1 
lon_res = 1

Latitude slice in degrees:
low=0
top=90

A file f"{low}to{top}".png will be created.

(reproduce.py) 
This script reproduces Figure 10 from the study:
doi:10.1029/2022JD038235


# ToDo 
•	Add detailed comments and clean up the code
•	Summarize key findings
•	Expand the analysis to explore:
•	Correlations between DMS (dimethyl sulfide) and sea surface area
•	2m surface temperature variations due to aerosol emissions

# Additional: download CMIP6 data
How to download CMIP6 data


(1) Read: https://pcmdi.llnl.gov/CMIP6/Guide/dataUsers.html and analyze and select data based on https://wcrp-cmip.github.io/CMIP6_CVs/docs/CMIP6_experiment_id.html

(2) Goto for instance https://esgf-node.llnl.gov/projects/cmip6/ and select the data. Add it to the chart and get the wget script. Make sure to be registered in the system before.

(3) Call the wget file by including -s flag to circumstance authoritisation.If you encounter permission errors, modify the file permissions by: chmod +x