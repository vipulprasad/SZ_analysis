# Code to stack cluster NILC maps
# The standard deviation map is taken as the map with maximum mean squared value

import healpy as hp
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
import numpy as np
from numpy import random
from astropy.io import fits
from astropy import units as u

# specify folders
output_folder = ""
input_folder = ""

cluster_catalog = fits.open(input_folder+ "HFI_PCCS_SZ-union_R2.08.fits")[1].data  
cluster_parameters = fits.open(input_folder+ "HFI_PCCS_SZ-union_R2.08.fits")[0].header
planck_nilc_stddev = hp.read_map(input_folder+ "COM_CompMap_Compton-SZMap-nilc-stddev_2048_R2.00.fits", hdu = 1)
planck_nilc_ymap = hp.read_map(input_folder+ "COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits", hdu = 1)
glon = cluster_catalog.GLON
glat = cluster_catalog.GLAT

num_clusters = len(cluster_catalog)
plot_size = 200

stack_array = np.zeros((plot_size,plot_size))
stack_stddev = np.zeros((plot_size,plot_size))
for i in range(num_clusters):
    crop_map = hp.gnomview(planck_nilc_ymap, rot=[glon[i], glat[i]], xsize = plot_size, ysize = plot_size, return_projected_map=True, no_plot=True).data
    crop_stddev_map = hp.gnomview(planck_nilc_stddev, rot=[glon[i], glat[i]], xsize = plot_size, ysize = plot_size, return_projected_map=True, no_plot=True).data
    if np.sum(crop_stddev_map**2) > np.sum(stack_stddev**2):
        stack_stddev = crop_stddev_map
    stack_array += crop_map

stack_ymap = stack_array/num_clusters
np.savetxt(output_folder+"stacked_nilc_map_planck.txt", stack_ymap)
np.savetxt(output_folder+"stacked_nilc_stddev_map_planck.txt", stack_stddev)
