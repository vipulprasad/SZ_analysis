import healpy as hp
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
import numpy as np
from numpy import random
from astropy.io import fits
from math import sqrt
import scipy.ndimage as ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cobaya.run import run
import getdist.plots as gdplt
import os
from scipy import stats
import pandas as pd

def show_image_line(images, figtitle = None, labels = [], output = None):

    num_images = len(images)
    if num_images == 1:
        plt.figure(figsize=(4,4))
        plt.title(labels[0])
        image =plt.imshow(images[0])
        plt.colorbar(image, orientation='horizontal', shrink=0.7, ticks = [np.min(images[0]), np.max(images[0])])
        plt.tight_layout()
    
    else:
        fig, axs = plt.subplots(1, num_images, figsize=(num_images*3,3))
        for i in range(num_images):
            ax=axs[i]
            if len(labels) != 0:
                ax.set_title(labels[i])
            image=ax.imshow(images[i])
            divider = make_axes_locatable(ax)
            cbar=plt.colorbar(image, orientation='horizontal', shrink=0.7, ticks = [np.min(images[i]), np.max(images[i])])

        fig.suptitle(figtitle, fontsize=16)
        fig.tight_layout()

    if output == None:
        plt.show()
    else:
        plt.savefig(output, dpi=300)

output_folder = "" #specify the path to output
input_folder = "" #specify the path to input files
cluster_name = "psz2g263.68-22.55" # "psz2g057.80+88.00" # "psz2g263.68-22.55"
cluster_catalog = fits.open(input_folder+ "HFI_PCCS_SZ-union_R2.08.fits")[1].data  
planck_nilc_stddev = hp.read_map(input_folder+ "COM_CompMap_Compton-SZMap-nilc-stddev_2048_R2.00.fits", hdu = 1)
planck_nilc_ymap = hp.read_map(input_folder+ "COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits", hdu = 1)



noise_scale = 1e3 # devise factor for noise to account for required sensitivity and effects of stacking
grid_size = 128 # grid size for cluster map

Ycl_val = 1e-4 # input cluster compton parameter for simulation
Yp_val = 1e-6 # input CMB y-type distortion parameter for simulation
Yp_prior = 3e-6 # mean value for the prior for y-type parameter (COBE result)
Yp_prior_stddev =11e-6 # standard deviation for the prior for the y-type parameter (COBE result)


## Creating directories

cluster_name_folder = cluster_name+"_(Ycl={:.1e},Yp={:.1e})".format(Ycl_val, Yp_val)

if not os.path.exists(output_folder+"/"+cluster_name_folder):
      os.mkdir(output_folder+"/"+cluster_name_folder )

if not os.path.exists(output_folder+"/"+cluster_name_folder+ "/cobaya_output/"):
      os.mkdir(output_folder+"/"+cluster_name_folder+ "/cobaya_output/")

cobaya_output_folder = output_folder+"/"+cluster_name_folder+ "/cobaya_output/"

plots_output_root = output_folder+"/"+cluster_name_folder + "/"+cluster_name +"_Planck_(Ycl={:.1e},Yp={:.1e})".format(Ycl_val, Yp_val)


plots_output_root = output_folder+"/"+cluster_name_folder + "/"+cluster_name

output_txtf = plots_output_root + "_Yp_analysis.txt"

f_output = open(output_txtf, "w")
f_output.write("__________________{}________________\n".format(cluster_name))

cluster_params = cluster_catalog[cluster_catalog.NAME == "PSZ2 G"+cluster_name.split('g')[-1]]
cluster_nilc_stddev = hp.gnomview(planck_nilc_stddev, rot=[cluster_params.GLON, cluster_params.GLAT], xsize = grid_size, ysize = grid_size, return_projected_map=True, no_plot=True).data
cluster_nilc_map = hp.gnomview(planck_nilc_ymap, rot=[cluster_params.GLON, cluster_params.GLAT], xsize = grid_size, ysize = grid_size, return_projected_map=True, no_plot=True).data


show_image_line([cluster_nilc_map, cluster_nilc_stddev], labels = ["NILC Map", "NILC std dev Map"],  figtitle = "psz2g263.68-22.55", output = plots_output_root+"NILC_map.png")

# Simulating the NILC map using an injected Ycl(cluster compton parameter) value

cluster_y_max = np.max(cluster_nilc_map)
cluster_nilc_map_normalized = cluster_nilc_map.data/cluster_y_max
cluster_nilc_stddev_rms = sqrt(np.mean(cluster_nilc_stddev**2)) # scalar (mean stddev of all pixels)
cluster_nilc_stddev_scaled = cluster_nilc_stddev/noise_scale

Ycl_map = (cluster_nilc_map_normalized*Ycl_val)
Ycl_map_noise = random.normal(0, scale = cluster_nilc_stddev_scaled) #, size = (grid_size,grid_size))

smooth_ang = 10 # arcmins
sigma_10arcmin = (smooth_ang) / (2. * np.sqrt(2. * np.log(2)))
final_Ycl_map = ndimage.gaussian_filter(Ycl_map, sigma=sigma_10arcmin, order=0) + Ycl_map_noise



show_image_line([cluster_nilc_map, cluster_nilc_stddev, final_Ycl_map, Ycl_map_noise], labels = ["NILC map", r"NILC std dev ($\sqrt{V}$)", r"Simulated Noise", r"Convolved $Map_{obs}$"], output = plots_output_root+"simulated.png", figtitle = cluster_name)
show_image_line([cluster_nilc_map, cluster_nilc_stddev], labels =["NILC map", r"NILC std dev($\sqrt{V}$)"], figtitle = "", output = plots_output_root+"NILC_planck.png")
show_image_line([final_Ycl_map, Ycl_map_noise], labels = [r"Noise", r"NILC $Map_{obs}$"],  figtitle = "Simulated Maps", output = plots_output_root+"simulated_maps_2.png")
show_image_line([Ycl_map_noise, cluster_nilc_stddev_scaled], labels = [r"Simulated Noise", r"NILC stddev / {}".format(noise_scale)],  figtitle = "Noise comparison", output = plots_output_root+"noise_comparison.png")
show_image_line([final_Ycl_map], labels =  [r"$Map_{Sim, NILC}$"], output = plots_output_root+"simulated_NILC_map.png")
show_image_line([cluster_nilc_stddev_scaled], labels = [r"$\sqrt{V}/1000$"], output = plots_output_root+"scaled_NILC_stddev_map.png")


def likelihood_Ycl(Ycl):
  # likelikhood = exp(- chi_squure)
  predicted_map = ndimage.gaussian_filter(Ycl*cluster_nilc_map_normalized, sigma=sigma_10arcmin, order=0)
  chisqr = np.sum(((final_Ycl_map - predicted_map)**2)/(cluster_nilc_stddev_scaled**2))/2
  return -chisqr # returning the log of likelihood

info={
      "likelihood": {"chisqr": likelihood_Ycl},
      "params":{"Ycl": {"prior":{"min":2e-5, "max":5e-3}, "ref":{"dist": 'norm', "loc":Ycl_val, "scale":Ycl_val/(10*noise_scale)}, "proposal":Ycl_val/(25*noise_scale)}},
      "sampler": {"mcmc": {"Rminus1_stop": 0.008, "max_tries": 100000, "learn_proposal":True}},
      "force": True
      }


print(cobaya_output_folder+cluster_name)

info["output"] = cobaya_output_folder +"{}_cobaya_output".format(cluster_name)

print(info)

updated_info_Ycl, sampler_Ycl = run(info)
updated_info_minimizer_Ycl, minimizer_Ycl = run(info, minimize=True)

print(minimizer_Ycl.products()["minimum"])

# Export the results to GetDist
gd_sample = sampler_Ycl.products(to_getdist=True, skip_samples=0.3)["sample"]

# Analyze and plot
Ycl_fit = gd_sample.getMeans(pars = [0])[0]
Ycl_fit_covariance = gd_sample.getCovMat().matrix
Ycl_fit_stddev = np.sqrt(Ycl_fit_covariance[0][0]) #/100

f_output.write("_____________Ycl result___________\n\n")
f_output.write(pd.DataFrame(info["params"]).to_string()+ "\n\n\n")
f_output.write("Mean = {}\n".format(Ycl_fit))
f_output.write("Std dev = {}\n\n\n\n".format(Ycl_fit_stddev))
print("Mean:")
print(Ycl_fit)
print("Covariance matrix:")
print(Ycl_fit_covariance, Ycl_fit_stddev)


gdplot = gdplt.get_subplot_plotter(subplot_size = 5)

gdplot.triangle_plot(gd_sample, ["Ycl"], filled=True, title_limit=1, fmt=".3E", labelsize = 14, markers = {"Ycl": 1e-4})
#g.set_params_format(fmt=".3E")
plt.savefig(plots_output_root + "Planck_Yp_analysis_ycl.pdf")
plt.close()


# Yp analysis

amplitude = Ycl_val*(Ycl_val + 2*Yp_val)


Y_yp_map = (cluster_nilc_map_normalized*amplitude)
np.random.seed(10)
Y_yp_map_noise = random.normal(0, scale = cluster_nilc_stddev_scaled)

Y_yp_map_convolve = ndimage.gaussian_filter(Y_yp_map, sigma=sigma_10arcmin, order=0)  + Y_yp_map_noise
# Y_yp_map_convolve = (final_Ycl_map + 2*Yp_val)*final_Ycl_map

show_image_line([Y_yp_map_convolve, Y_yp_map_noise], labels = [r"Noise", r"CILC $Map_{obs}$"],  figtitle = "Simulated Maps", output = plots_output_root+"simulated_maps_Yp_2.png")
show_image_line([Y_yp_map_convolve], labels = [r"$Map_{Sim, CILC}$"], output = plots_output_root+"simulated_CILC_map.png")


def likelihood_Ycl_Yp(Ycl, Yp):

    Ycl_map = Ycl*(2*Yp +Ycl*cluster_nilc_map_normalized)*cluster_nilc_map_normalized
    map_prediction = ndimage.gaussian_filter(Ycl_map, sigma=sigma_10arcmin, order=0)
    chisqr = np.sum(((Y_yp_map_convolve - map_prediction)**2)/(cluster_nilc_stddev_scaled)**2)/2
    return -chisqr # returning log(likelihood)

a_yp_prior = -Yp_prior/Yp_prior_stddev
b_yp_prior = (1 + Yp_prior)/Yp_prior_stddev

info_Yp={
      "likelihood": {"chisqr": likelihood_Ycl_Yp},
      "params":{
                "Ycl": {"prior": {"dist": 'norm', "loc":Ycl_fit, "scale":Ycl_fit_stddev}, "ref": {"dist": 'norm', "loc":Ycl_val, "scale":Ycl_fit_stddev}, "proposal":Ycl_fit_stddev},
                 "Yp": {"prior": {"dist": 'truncnorm', "a": a_yp_prior, "b": b_yp_prior, "loc":Yp_prior, "scale":Yp_prior_stddev}, "proposal":Yp_prior_stddev/1e2}
                },
      "sampler": {"mcmc": {"Rminus1_stop": 0.008, "max_tries": 100000, "learn_proposal":True}},
      "force": True
      }

info_Yp["output"] = cobaya_output_folder+ "{}_Yp_cobaya_output".format(cluster_name)


print(info_Yp)

updated_info_Ycl_Yp, sampler_Ycl_Yp = run(info_Yp)
updated_info_minimizer_Ycl_Yp, minimizer_Ycl_Yp = run(info_Yp, minimize=True)

# Export the results to GetDist
gd_sample_Yp = sampler_Ycl_Yp.products(skip_samples=0.3, to_getdist=True)["sample"]
# Analyze and plot
fit_values = gd_sample_Yp.getMeans()
Ycl_Yp_covmat = gd_sample_Yp.getCovMat().matrix
Yp_fit = fit_values[1]
Yp_fit_stddev = sqrt(Ycl_Yp_covmat[1][1])
print("Mean:")
print(Yp_fit)
print("Covariance matrix:")
print(Ycl_Yp_covmat)

# upper limits

Yp_upper_limit_2sigma = gd_sample_Yp.confidence(1, 0.05, upper = True)
Yp_upper_limit_1sigma = gd_sample_Yp.confidence(1, 0.3173, upper = True)


f_output.write("_____________Yp result___________\n\n")
f_output.write(pd.DataFrame(info_Yp["params"]).to_string()+ "\n\n\n")
f_output.write("Mean Ycl, Yp = {}, {}\n".format(fit_values[0], fit_values[1]))
f_output.write("Cov Mat:\n")
f_output.write("{} \n\n".format(Ycl_Yp_covmat))
f_output.write("Upper limits for Yp\n")
f_output.write("Yp < {} (1 sigma)\n".format(Yp_upper_limit_1sigma))
f_output.write("Yp < {} (2 sigma)\n".format(Yp_upper_limit_2sigma))


x_yp = np.linspace(Yp_prior-4*Yp_prior_stddev, Yp_prior+4*Yp_prior_stddev, 1000)
yp_prior_vals = stats.norm.pdf(x_yp, loc = Yp_prior, scale = Yp_prior_stddev)

Ycl_limit1, Ycl_limit2 = Ycl_fit - 6*Ycl_fit_stddev, Ycl_fit + 6*Ycl_fit_stddev
yp_limit1, yp_limit2 = Yp_val - Yp_prior_stddev, Yp_val + Yp_prior_stddev

x_Ycl = np.linspace(Ycl_fit - 4*Ycl_fit_stddev, Ycl_fit + 4*Ycl_fit_stddev, 1000)
Ycl_prior_vals = stats.norm.pdf(x_Ycl, loc = Ycl_fit, scale = Ycl_fit_stddev)

gdplot = gdplt.get_subplot_plotter(subplot_size = 4)
#gdplot = gdplt.GetDistPlotter()
gdplot.triangle_plot([gd_sample_Yp], ["Ycl", "Yp"], filled=True, legend_labels = ['Posteriors'], title_limit=1, fontsize= 16, labelsize=14, fmt=".3E", markers = {"Ycl": Ycl_val, "Yp": Yp_val}, rc_sizes = 40)#, param_limits =  {"Ycl": (Ycl_limit1, Ycl_limit2), "Yp": (yp_limit1, yp_limit2)}) # 68% percentage limit

ax_11 = gdplot.get_axes(('Ycl',))
ax_12 = gdplot.get_axes(('Yp',))

ax_11.plot(x_Ycl, Ycl_prior_vals/np.max(Ycl_prior_vals), label = "Ycl prior", color = 'red', alpha = 0.5)
ax_11.legend()
ax_12.plot(x_yp, yp_prior_vals/np.max(yp_prior_vals), label = "Yp prior", color = 'red', alpha = 0.5)
ax_12.legend()
plt.tight_layout()
plt.savefig(plots_output_root +"_Planck_Yp_analysis_ycl&Yp.pdf")
plt.close()

# Yp 2nd analysis 


info_Yp_2={
      "likelihood": {"chisqr": likelihood_Ycl_Yp},
      "params":{
                "Ycl": {"prior": {"dist": 'norm', "loc":Ycl_fit, "scale":Ycl_fit_stddev}, "ref": {"dist": 'norm', "loc":Ycl_val, "scale":Ycl_fit_stddev}, "proposal":Ycl_fit_stddev},
                "Yp": {"prior": {"dist": 'norm', "loc":Yp_prior, "scale":Yp_prior_stddev}, "ref": {"dist": 'norm', "loc":Yp_fit, "scale": Yp_fit_stddev}, "proposal":Yp_prior_stddev/1e3}
                },
      "sampler": {"mcmc": {"Rminus1_stop": 0.004, "max_tries": 100000, "learn_proposal":True}},
      "force": True
      }

info_Yp_2["output"] = cobaya_output_folder+ "{}_Yp_2_cobaya_output".format(cluster_name)


print(info_Yp_2)

updated_info_Ycl_Yp_2, sampler_Ycl_Yp_2 = run(info_Yp_2)
updated_info_minimizer_Ycl_Yp_2, minimizer_Ycl_Yp_2 = run(info_Yp_2, minimize=True)

# Export the results to GetDist
gd_sample_Yp_2 = sampler_Ycl_Yp_2.products(skip_samples=0.3, to_getdist=True)["sample"]
# Analyze and plot
fit_values_2 = gd_sample_Yp_2.getMeans()
Ycl_Yp_2_covmat = gd_sample_Yp.getCovMat().matrix
Yp_fit_2 = fit_values_2[1]
print("Mean:")
print(Yp_fit_2)
print("Covariance matrix:")
print(Ycl_Yp_2_covmat)

# upper limits

Yp_upper_limit_2sigma_2 = gd_sample_Yp.confidence(1, 0.05, upper = True)
Yp_upper_limit_1sigma_2 = gd_sample_Yp.confidence(1, 0.3173, upper = True)


f_output.write("_____________Yp 2nd analysis result___________\n\n")
f_output.write(pd.DataFrame(info_Yp_2["params"]).to_string()+ "\n\n\n")
f_output.write("Mean Ycl, Yp = {}, {}\n".format(fit_values_2[0], fit_values_2[1]))
f_output.write("Cov Mat:\n")
f_output.write("{} \n\n".format(Ycl_Yp_2_covmat))
f_output.write("Upper limits for Yp\n")
f_output.write("Yp < {} (1 sigma)\n".format(Yp_upper_limit_1sigma))
f_output.write("Yp < {} (2 sigma)\n".format(Yp_upper_limit_2sigma))


x_yp = np.linspace(Yp_prior-4*Yp_prior_stddev, Yp_prior+4*Yp_prior_stddev, 1000)
yp_prior_vals = stats.norm.pdf(x_yp, loc = Yp_prior, scale = Yp_prior_stddev)

Ycl_limit1, Ycl_limit2 = Ycl_fit - 6*Ycl_fit_stddev, Ycl_fit + 6*Ycl_fit_stddev
yp_limit1, yp_limit2 = Yp_val - Yp_prior_stddev, Yp_val + Yp_prior_stddev

x_Ycl = np.linspace(Ycl_fit - 4*Ycl_fit_stddev, Ycl_fit + 4*Ycl_fit_stddev, 1000)
Ycl_prior_vals = stats.norm.pdf(x_Ycl, loc = Ycl_fit, scale = Ycl_fit_stddev)

gdplot = gdplt.get_subplot_plotter(subplot_size = 4, rc_sizes=16, scaling=True)
#gdplot = gdplt.GetDistPlotter()
gdplot.triangle_plot([gd_sample_Yp_2], ["Ycl", "Yp"], filled=True, legend_labels = ['Posteriors'], title_limit=1, fontsize= 16, markers = {"Ycl": Ycl_val, "Yp": Yp_val}, rc_sizes = 40)#, param_limits =  {"Ycl": (Ycl_limit1, Ycl_limit2), "Yp": (yp_limit1, yp_limit2)}) # 68% percentage limit

ax_11 = gdplot.get_axes(('Ycl',))
ax_12 = gdplot.get_axes(('Yp',))

ax_11.plot(x_Ycl, Ycl_prior_vals/np.max(Ycl_prior_vals), label = "Ycl prior", color = 'red', alpha = 0.5)
ax_11.legend()
ax_12.plot(x_yp, yp_prior_vals/np.max(yp_prior_vals), label = "Yp prior", color = 'red', alpha = 0.5)
#ax_12.set_xlim((0, 5e-6))
ax_12.legend()
plt.tight_layout()
plt.savefig(plots_output_root + "_Planck_Yp_analysis_ycl&Yp_2nd_run.pdf")
plt.close()

f_output.close()
