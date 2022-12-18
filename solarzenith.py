"""
DEMO TO COMPUTE SOLAR ZENITH ANGLE
version 6 April 2017
by Antti Lipponen
Copyright (c) 2017 Antti Lipponen
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np  # numerics & matrix algebra
from datetime import datetime  # dates and times
import time  # for measuring time
import matplotlib as mpl  # plotting
import matplotlib.pyplot as plt  # plotting
from mpl_toolkits.basemap import Basemap  # map

#################################################################################
# USER GIVEN PARAMETERS

# time
t = datetime(2017, 4, 6, 9, 30, 0)  # 6th April 2017 09:30:00 UTC

# grid dimensions
Nlats = 90
Nlons = 180

#################################################################################


#################################################################################
# FUNCTION TO COMPUTE SOLAR AZIMUTH AND ZENITH ANGLE
# translated to Python from http://www.psa.es/sdg/sunpos.htm
#################################################################################
def szafunc(day, dLongitude, dLatitude):
    """
        inputs: day: datetime object
                dLongitude: longitudes (scalar or Numpy array)
                dLatitude: latitudes (scalar or Numpy array)
        output: solar zenith angles
    """
    dHours, dMinutes, dSeconds = day.hour, day.minute, day.second
    iYear, iMonth, iDay = day.year, day.month, day.day

    dEarthMeanRadius = 6371.01
    dAstronomicalUnit = 149597890

    ###################################################################
    # Calculate difference in days between the current Julian Day
    # and JD 2451545.0, which is noon 1 January 2000 Universal Time
    ###################################################################
    # Calculate time of the day in UT decimal hours
    dDecimalHours = dHours + (dMinutes + dSeconds / 60.) / 60.
    # Calculate current Julian Day
    liAux1 = int((iMonth - 14.) / 12.)
    liAux2 = int((1461. * (iYear + 4800. + liAux1)) / 4.) + int((367. * (iMonth - 2. - 12. * liAux1)) / 12.) - int((3. * int((iYear + 4900. + liAux1) / 100.)) / 4.) + iDay - 32075.
    dJulianDate = liAux2 - 0.5 + dDecimalHours / 24.
    # Calculate difference between current Julian Day and JD 2451545.0
    dElapsedJulianDays = dJulianDate - 2451545.0

    ###################################################################
    # Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
    # ecliptic in radians but without limiting the angle to be less than 2*Pi
    # (i.e., the result may be greater than 2*Pi)
    ###################################################################
    dOmega = 2.1429 - 0.0010394594 * dElapsedJulianDays
    dMeanLongitude = 4.8950630 + 0.017202791698 * dElapsedJulianDays  # Radians
    dMeanAnomaly = 6.2400600 + 0.0172019699 * dElapsedJulianDays
    dEclipticLongitude = dMeanLongitude + 0.03341607 * np.sin(dMeanAnomaly) + 0.00034894 * np.sin(2. * dMeanAnomaly) - 0.0001134 - 0.0000203 * np.sin(dOmega)
    dEclipticObliquity = 0.4090928 - 6.2140e-9 * dElapsedJulianDays + 0.0000396 * np.cos(dOmega)

    ###################################################################
    # Calculate celestial coordinates ( right ascension and declination ) in radians
    # but without limiting the angle to be less than 2*Pi (i.e., the result may be
    # greater than 2*Pi)
    ###################################################################
    dSin_EclipticLongitude = np.sin(dEclipticLongitude)
    dY = np.cos(dEclipticObliquity) * dSin_EclipticLongitude
    dX = np.cos(dEclipticLongitude)
    dRightAscension = np.arctan2(dY, dX)
    if dRightAscension < 0.0:
        dRightAscension = dRightAscension + 2.0 * np.pi
    dDeclination = np.arcsin(np.sin(dEclipticObliquity) * dSin_EclipticLongitude)

    ###################################################################
    # Calculate local coordinates ( azimuth and zenith angle ) in degrees
    ###################################################################
    dGreenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * dElapsedJulianDays + dDecimalHours
    dLocalMeanSiderealTime = (dGreenwichMeanSiderealTime * 15. + dLongitude) * (np.pi / 180.)
    dHourAngle = dLocalMeanSiderealTime - dRightAscension
    dLatitudeInRadians = dLatitude * (np.pi / 180.)
    dCos_Latitude = np.cos(dLatitudeInRadians)
    dSin_Latitude = np.sin(dLatitudeInRadians)
    dCos_HourAngle = np.cos(dHourAngle)
    dZenithAngle = (np.arccos(dCos_Latitude * dCos_HourAngle * np.cos(dDeclination) + np.sin(dDeclination) * dSin_Latitude))
    dY = -np.sin(dHourAngle)
    dX = np.tan(dDeclination) * dCos_Latitude - dSin_Latitude * dCos_HourAngle
    dAzimuth = np.arctan2(dY, dX)
    dAzimuth[dAzimuth < 0.0] = dAzimuth[dAzimuth < 0.0] + 2.0 * np.pi
    dAzimuth = dAzimuth / (np.pi / 180.)
    # Parallax Correction
    dParallax = (dEarthMeanRadius / dAstronomicalUnit) * np.sin(dZenithAngle)
    dZenithAngle = (dZenithAngle + dParallax) / (np.pi / 180.)

    return dAzimuth - 180.0, dZenithAngle


#################################################################################
# COMPUTE ZENITH ANGLES AND AZIMUTHS
#################################################################################

# coordinates in grid
lat, lon = np.linspace(-90.0, 90.0, Nlats + 1), np.linspace(-180.0, 180.0, Nlons + 1)  # lat and lon vectors for grid boundaries
latC, lonC = 0.5 * (lat[:-1] + lat[1:]), 0.5 * (lon[:-1] + lon[1:])  # center points

# make grid
latgrid, longrid = np.meshgrid(latC, lonC)

t0 = time.time()  # measure time to compute szas
# compute solar zenith angle and azimuth (be careful with azimuth: I haven't checked this at all)
saz, sza = szafunc(t, longrid.ravel(), latgrid.ravel())
print('Computed {} solar zenith angles and azimuths and it took {:.04f} seconds'.format(len(longrid.ravel()), time.time() - t0))


#################################################################################
# PLOT RESULTS
#################################################################################

# get colormap viridis (http://matplotlib.org/examples/color/colormaps_reference.html)
cmap = mpl.cm.viridis

# save colorbar to separate figure (solar zenith angle)
fig = plt.figure(figsize=(14, 8))
axCB = fig.add_axes([0.05, 0.10, 0.90, 0.03])
cb1 = mpl.colorbar.ColorbarBase(axCB, cmap=cmap, norm=mpl.colors.Normalize(vmin=0.0, vmax=90.0), orientation='horizontal')
cb1.set_label('Solar zenith angle')
plt.savefig('colormapSZA.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()

# save colorbar to separate figure (solar azimuth angle)
fig = plt.figure(figsize=(14, 8))
axCB = fig.add_axes([0.05, 0.10, 0.90, 0.03])
cb1 = mpl.colorbar.ColorbarBase(axCB, cmap=cmap, norm=mpl.colors.Normalize(vmin=-180.0, vmax=180.0), orientation='horizontal')
cb1.set_label('Solar azimuth angle')
plt.savefig('colormapSAZ.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()


# plot map of solar zenith angles
fig = plt.figure(figsize=(14, 8), frameon=False)
ax = fig.add_axes([0.05, 0.10, 0.9, 0.9])

# cylindrical projection
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
m.drawmeridians(np.arange(-180., 181., 30.), linewidth=0.5, color='#A0A0A0')
m.drawparallels(np.arange(-90., 91., 30.), linewidth=0.5, color='#A0A0A0')

x, y = m(longrid, latgrid)  # convert to map projection coordinates
contour = m.contourf(x, y, sza.reshape(Nlons, Nlats), np.linspace(0.0, 90.0, 90.0), cmap=cmap)
plt.title('Solar zenith angle, {}'.format(t.strftime('%d.%m.%Y %H:%M:%S UTC')))
plt.savefig('SZAmap.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()


# plot map of solar azimuth angles
fig = plt.figure(figsize=(14, 8), frameon=False)
ax = fig.add_axes([0.05, 0.10, 0.9, 0.9])

# cylindrical projection
m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
m.drawmeridians(np.arange(-180., 181., 30.), linewidth=0.5, color='#A0A0A0')
m.drawparallels(np.arange(-90., 91., 30.), linewidth=0.5, color='#A0A0A0')

x, y = m(longrid, latgrid)  # convert to map projection coordinates
contour = m.contourf(x, y, saz.reshape(Nlons, Nlats), np.linspace(-180.0, 180.0, 360.0), cmap=cmap)
plt.title('Solar azimuth angle, {}'.format(t.strftime('%d.%m.%Y %H:%M:%S UTC')))
plt.savefig('SAZmap.png', dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()

# and we're done!