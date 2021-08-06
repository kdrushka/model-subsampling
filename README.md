## model-subsampling
## Continuation of OceanHackWeek21 project to subsample high-resolution model output as if by gliders, ships, or other in situ platforms

The goal of this project is to create a Python package that takes an input trajectory (e.g., the path of an ocean glider), subsamples output from a high-resolution ocean simulation along that trajectory, and returns a set of subsampled variables (e.g., standard physical variables temperature, salinity, velocity; derived physical quantities such as steric height; biogeochemical quantities if available).  We envision this package having two potential uses: 1) designing *in situ* sampling strategies, and 2) interpreting *in situ* data in the context of a highly resolved oceanographic model.

Tasks:
1. Load model data from a specific region (e.g., ROMS output in the California Current from CenOOS; MITgcm regional llc4320 data)
2. Create simulated trajectory file
3. Obtain real trajectory file
4. Interpolate model data to the trajectories
5. Compare interpolated data to the full model output via useful visualizations and/or statistics


Notes:
- FinalProjectSummary.ipynb contains the final project notebook developed during OHW21
- Individual notebooks that went into FinalProjectSummary.ipynb are in the OHW21_code folder 
