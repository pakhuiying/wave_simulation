# Sun glitter simulation

This python script and notebook generates sun glitter distribution on Cox-Munk random sea surface slopes. Daughter rays are saved to inspect more details on the scattered daughter rays and intercepted wave facets (if user specifies `save_fp` in the notebook/command line). Examples of saved daughter rays are provided.

### Python Notebook 
The notebook `Cox_Munk_wave.ipynb` contains a step-by-step guide and mathematical theory on constructing a random sea surface slope to ray tracing to the simulation of glitter on the cox-munk surface. Do follow this guide for a systematic procedure on sun glitter simulation. The figures and theory in the notebook are obtained from *Preisendorfer and Mobley (1985)*: 

Preisendorfer, R. W., & Mobley, C. D. (1985). Unpolarized Irradiance Reflectances and Glitter Patterns of Random Capillary Waves on Lakes and Seas. Monte Carlo Simulation. NOAA Technical Memorandum ERL PMEL-63.

### Command-line interface
Alternatively, one can also run the python script for command-line interface, where user can specify:

- `--solar_altitude`: solar_altitude (angle in deg from the horizontal)
- `--solar_azimuth`: solar_azimuth (angle in deg from the i axis (along wind direction))
- `--wind_speed`: wind speed in m/s
- `--n`: order of hexagonal domain
- `--iter`: number of parent rays to seed
- `--save_fp`: directory on where to save the daughter rays
- `--camera_altitude`: camera_altitude (angle in deg from the horizontal)
- `--camera_azimuth`: camera_azimuth (angle in deg from the i axis (along wind direction))'

See `-h` for help and details on input

## Output

![glitter pattern](images/output.png 'glitter pattern simulation')