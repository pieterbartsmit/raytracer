import raytracer
import numpy
import spectra
import matplotlib.pyplot as plt
import scipy.interpolate

#-------------------
# SETUP BATHYMETRY:
#-------------------
lat = numpy.array([1,2])
lon = numpy.array([0,1])

# Note dep is ordered as dep[lat,lon] in python
dep = numpy.array([[ 1000, 1 ],
       [ 1000, 1 ]])

#-------------------------
# Setup calculation domain
#-------------------------
#
# origin
origin_lat_lon = (0.,0)  # (lat,lon)

# Domain width
lat_width = 3
lon_width = 1

# where the incident wave boundary is located
boundary = ['e'] # List, if there are multiply incident boundaries, just add


rad2deg = 180./numpy.pi

#-------------------------
# Numerics
#-------------------------
# number of subrays.
nsub = 100

# create raytracer object
rt = raytracer.RayTracer( origin_lat_lon , lat_width, lon_width, lat, lon, dep,
        bndlist = boundary,dcont=0.1)

#-------------------------
# Output
#-------------------------
# Points where we want output, just two numpy arrays of lats and lons.
outputline_lon = numpy.linspace( 0,1,11,endpoint=True )
outputline_lat = numpy.zeros(outputline_lon.shape) + 1.5

#-------------------------
# Spectral settings
#-------------------------
# Number of frequencies and directions we are calculating with. directions
# should span the circle. Here I use a single frequency (always may it an array)
# Frequencies are radial frequencies, directions are in radians.
frequencies = numpy.linspace( 0.02,0.02,1,endpoint=True) * numpy.pi * 2
directions  = numpy.linspace( -180,180,36,endpoint=False) * numpy.pi / 180.


#-------------------------
# Create transfer matrix
#-------------------------
#
# The transfer matrix relates a spectrum at the desired points to the incident
# spectrum at the boundary
mat = rt.getmatrix( outputline_lon,outputline_lat, directions, nsub, frequencies)

#-------------------------
# Estimate spectra at points
#-------------------------

# this is just a raised cosins distribution in directional space
dirspec = spectra.raised_cosine( directions * rad2deg, mean_direction=-180, width=100 )*rad2deg
print(numpy.trapz(dirspec,directions))

# incident spectrum has a single frequency (hence the none to make array 2d)
incident_hs = 0.1
incident_spectrum = incident_hs**2 / 16 * dirspec[:,None]

# output spectra: dimension is [ number_of_output_points,ndir,nfreq ]
out = rt.prediction(frequencies,directions,incident_spectrum,outputline_lon,outputline_lat,nsub)

# ============================
# Some plots
# ============================
numpoint = len(outputline_lon)
hs = numpy.zeros( (numpoint,))
for ipoint in range( 0, numpoint):

    m0 = numpy.sum( out[ipoint,:,0]) * (directions[2]-directions[1])
    hs[ipoint] = 4 * numpy.sqrt(m0)
    plt.plot(directions,out[ipoint,:,0],label=str(ipoint))
plt.yscale('log')
plt.legend()

plt.figure()

# get interpolated depth
dep_intp = scipy.interpolate.interp1d( lon,dep[0,:] )
depi = dep_intp( outputline_lon )

# Energy should roughly scale as 1/d - check if the log plots look similar
plt.plot(depi,hs**2)
plt.plot(depi,1/depi)
#plt.plot(outputline_lon,1/depi)
plt.yscale('log')
plt.show()
