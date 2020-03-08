#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:52:39 2020

@author: Georgina Dransfield
"""



import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




def app_mag_err(star_err, flux_ratio_err, flux_ratio):
    i=0
    size = len(flux_ratio)
    errors = ([])
    while i<size:
        if flux_ratio_err[i]!=-9.9:
            error = np.sqrt(star_err[i]**2 + ((flux_ratio_err[i]*-2.5)/(flux_ratio[i]*np.log(10)))**2)
            errors.append(error)
            i+=1
        else:
            error = 0
            errors.append(error)
            i+=1
    nperrors = np.array(errors)
    return nperrors

def planet_mags(fname):
    planet_data = np.genfromtxt(fname, comments = '#', unpack = True)
    planet_names = np.genfromtxt(fname, comments = '#', unpack = True, dtype=str)
    star_names = planet_names[0]

    BJ_distance = planet_data[15]


    global J_rad
    J_rad = 69911000
    
    planet_radius = planet_data[31]*J_rad

    #Apparent magnitudes calculated for the exoplanets in the sample

    app_mags_pl_J = np.array(planet_data[1] - 2.5*np.log10(planet_data[17]/100))
    app_mags_pl_H = np.array(planet_data[3] - 2.5*np.log10(planet_data[19]/100))
    app_mags_pl_K = np.array(planet_data[5] - 2.5*np.log10(planet_data[21]/100))
    app_mags_pl_36 = np.array(planet_data[7] - 2.5*np.log10(planet_data[23]/100))
    app_mags_pl_45 = np.array(planet_data[9] - 2.5*np.log10(planet_data[25]/100))
    app_mags_pl_58 = np.array(planet_data[11] - 2.5*np.log10(planet_data[27]/100))
    app_mags_pl_80 = np.array(planet_data[13] - 2.5*np.log10(planet_data[29]/100))
    app_mags_pl_W = np.array(planet_data[33] - 2.5*np.log10(planet_data[35]/100))
    app_mags_pl_NB1 = np.array(planet_data[37] - 2.5*np.log10(planet_data[39]/100))
    app_mags_pl_NB2 = np.array(planet_data[41] - 2.5*np.log10(planet_data[43]/100))
    app_mags_pl_z = np.array(planet_data[45] - 2.5*np.log10(planet_data[47]/100))

    #Absolute magnitudes calculated for the exoplanets in the sample

    abs_mags_pl_J = np.array(app_mags_pl_J - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_H = np.array(app_mags_pl_H - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_K = np.array(app_mags_pl_K - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_36 = np.array(app_mags_pl_36 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_45 = np.array(app_mags_pl_45 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_58 = np.array(app_mags_pl_58 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_80 = np.array(app_mags_pl_80 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_W = np.array(app_mags_pl_W - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_NB1 = np.array(app_mags_pl_NB1 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_NB2 = np.array(app_mags_pl_NB2 - (5*np.log10(BJ_distance)) + 5)
    abs_mags_pl_z = np.array(app_mags_pl_z - (5*np.log10(BJ_distance)) + 5)
    

    abs_mags_pl = {'J': abs_mags_pl_J, 'H': abs_mags_pl_H, 'K': abs_mags_pl_K, '3.6': abs_mags_pl_36, \
                   '4.5': abs_mags_pl_45, '5.8': abs_mags_pl_58, '8.0': abs_mags_pl_80, 'W': abs_mags_pl_W,
                   'NB1190': abs_mags_pl_NB1, 'NB2090': abs_mags_pl_NB2, 'z': abs_mags_pl_z}
    
    global dwarf_rad
 
    dwarf_rad = 0.9*69911000
    
    

    adjJ = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_J)
    adjH = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_H)
    adjK = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_K)
    adj36 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_36)
    adj45 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_45)
    adj58 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_58)
    adj80 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_80)
    adjW = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_W)
    adjNB1 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_NB1)
    adjNB2 = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_NB2)
    adjz = np.array(((1/0.4)*np.log10(((planet_radius)**2)/(dwarf_rad)**2))+abs_mags_pl_z)

    adj_pl = {'J': adjJ, 'H': adjH, 'K': adjK, '3.6': adj36, \
              '4.5': adj45, '5.8': adj58, '8.0': adj80, 'W':  adjW,
              'NB1190':  adjNB1, 'NB2090':  adjNB2, 'z':  adjz}


    app_mags_errs_J = app_mag_err(planet_data[2], planet_data[18], planet_data[17])
    app_mags_errs_H = app_mag_err(planet_data[4], planet_data[20], planet_data[19])
    app_mags_errs_K = app_mag_err(planet_data[6], planet_data[22], planet_data[21])
    app_mags_errs_36 = app_mag_err(planet_data[8], planet_data[24], planet_data[23])
    app_mags_errs_45 = app_mag_err(planet_data[10], planet_data[26], planet_data[25])
    app_mags_errs_58 = app_mag_err(planet_data[12], planet_data[28], planet_data[27])
    app_mags_errs_80 = app_mag_err(planet_data[14], planet_data[30], planet_data[29])
    app_mags_errs_W = app_mag_err(planet_data[34], planet_data[36], planet_data[35])
    app_mags_errs_NB1 = app_mag_err(planet_data[38], planet_data[40], planet_data[38])
    app_mags_errs_NB2 = app_mag_err(planet_data[42], planet_data[44], planet_data[43])
    app_mags_errs_z = app_mag_err(planet_data[46], planet_data[48], planet_data[47])

    pl_errs = {'J': app_mags_errs_J, 'H': app_mags_errs_H, 'K': app_mags_errs_K, '3.6': app_mags_errs_36, \
               '4.5': app_mags_errs_45, '5.8': app_mags_errs_58, '8.0': app_mags_errs_80, 'W': app_mags_errs_W, 
               'NB1190': app_mags_errs_NB1, 'NB2090': app_mags_errs_NB2, 'z': app_mags_errs_z}
    h20 = planet_data[48]
    na = planet_data[49]
    k = planet_data[50]
    tio = planet_data[60]
    vo = planet_data[59]
    co = planet_data[62]
    
    chem = {'Water': h20, 'Na': na, 'K': k, 'TiO':tio, 'VO':vo, 'CO': co}
    
    
    return abs_mags_pl, adj_pl, pl_errs, chem, star_names


def remove_empty(band):
    clean = []
    for i in band:
        if i == -9999:
            j = np.log10(i)
            clean.append(j)
        else:
            j = i
            clean.append(j)
    clean = np.array(clean)
    return clean


#Spectral types in the file are given in their normal form, as a letter (to represent the type) followed by a floating point
#number (to represent the sub-type). The code in this cell converts Spectral types into floating point numbers with M=1, L=2, 
#T=3, and Y=4. The sub type is represented by a decimal. This allows the stars to be colormapped according to spectral type
#on the Colour Magnitude Diagrams later on.

#In order for this part of the code to work, please ensure that '.0' is added to all spectral types which don't end in '.5'.
#For example, 'M6' should be changed to 'M6.0'


#Creates two lists: one of string spectral types in their normal form called 'spec_types' and another with the numerical form
#called 'numbers'. Both lists are in order. 
spec_types = ['M', 'L', 'T', 'Y']
classes = len(spec_types)
sub_types = np.arange(0, 10, 0.5)
spec_types = [sptype + str(x) for sptype in spec_types for x in np.arange(0,10,0.5)]
numbers = [(i+1)+x/10 for i in range(classes) for x in sub_types]


#This function loops through a list of spectral classes as pulled in from the data and checks to see if it matches one of the 
#classes created above. If so, it allocates the corresponding number. 
def spectral_class(spec_class):    
    spec_classes_new = ([])
    for each in spec_class:
        if each == '-99.9':
            spec_class_new = 10.0
            spec_classes_new.append(spec_class_new)
        else:
            b=0
            while b < len(spec_types):
                if each == spec_types [b]:
                    spec_class_new = numbers [b]
                    spec_classes_new.append(spec_class_new)
                    break
                else:
                    b+=1
    return spec_classes_new

#Calculates Absolute magnitudes for the brown dwarf sample using imported distance moduli and apparent magnitudes. 

def dwarf_abs_mag(distance_modulus, app_mag):
    i = 0
    abs_mags = ([])
    for each in app_mag:
        if each != -99.9:
            abs_mag = each - distance_modulus[i]
        else:
            abs_mag = np.log10(each)
        abs_mags.append(abs_mag)
        i+=1
    abs_mags = np.array(abs_mags)
    return abs_mags

def brown_dwarfs(bd_fname):


    BD = np.genfromtxt(bd_fname, comments = '#', unpack = True, dtype = float)
    BD1 = np.genfromtxt(bd_fname, comments = '#', unpack = True, dtype = str)
    spectral_types = BD1[1]
    distance_moduli = BD[2]
    distance_moduli_errs = BD[3]
    app_mags_dwarves_36 = BD[10]
    app_mags_dwarves_36_err = np.array(BD[11]*0.01)
    app_mags_dwarves_45 = BD[12]
    app_mags_dwarves_45_err = np.array(BD[13]*0.01)
    app_mags_dwarves_58 = BD[14]
    app_mags_dwarves_58_err = np.array(BD[15]*0.01)
    app_mags_dwarves_80 = BD[16]
    app_mags_dwarves_80_err = np.array(BD[17]*0.01)
    app_mags_dwarves_J = BD[4]
    app_mags_dwarves_J_err = np.array(BD[5]*0.01)
    app_mags_dwarves_H = BD[6]
    app_mags_dwarves_H_err = np.array(BD[7]*0.01)
    app_mags_dwarves_K = BD[8]
    app_mags_dwarves_K_err = np.array(BD[9]*0.01) 



    app_mags_dwarves_36 = remove_empty (app_mags_dwarves_36)
    app_mags_dwarves_45 = remove_empty (app_mags_dwarves_45)
    app_mags_dwarves_58 = remove_empty (app_mags_dwarves_58)
    app_mags_dwarves_80 = remove_empty (app_mags_dwarves_80)

    app_mags_dwarves_J = remove_empty (app_mags_dwarves_J)
    app_mags_dwarves_H = remove_empty (app_mags_dwarves_H)
    app_mags_dwarves_K = remove_empty (app_mags_dwarves_K)

    app_mags_dwarves_36_err = remove_empty (app_mags_dwarves_36_err)
    app_mags_dwarves_45_err = remove_empty (app_mags_dwarves_45_err)
    app_mags_dwarves_58_err = remove_empty (app_mags_dwarves_58_err)
    app_mags_dwarves_80_err = remove_empty (app_mags_dwarves_80_err)

    app_mags_dwarves_J_err = remove_empty (app_mags_dwarves_J_err)
    app_mags_dwarves_H_err = remove_empty (app_mags_dwarves_H_err)
    app_mags_dwarves_K_err = remove_empty (app_mags_dwarves_K_err)
    


    dwarf_errs = {'J': app_mags_dwarves_J_err, 'H': app_mags_dwarves_H_err, 'K': app_mags_dwarves_K_err, \
              '3.6': app_mags_dwarves_36_err, '4.5': app_mags_dwarves_45_err, '5.8': app_mags_dwarves_58_err, \
              '8.0': app_mags_dwarves_80_err}



    spectral_classes = np.array(spectral_class(spectral_types))


    SpT = {'J': spectral_classes, 'H': spectral_classes, 'K': spectral_classes, '3.6': spectral_classes, \
       '4.5': spectral_classes, '5.8': spectral_classes, '8.0': spectral_classes}



    abs_mags_dwarves_J = np.array(dwarf_abs_mag(distance_moduli, app_mags_dwarves_J))
    abs_mags_dwarves_H = np.array(dwarf_abs_mag(distance_moduli, app_mags_dwarves_H))
    abs_mags_dwarves_K = np.array(dwarf_abs_mag(distance_moduli, app_mags_dwarves_K))
    abs_mags_dwarves_36 = dwarf_abs_mag(distance_moduli, app_mags_dwarves_36)
    abs_mags_dwarves_45 = dwarf_abs_mag(distance_moduli, app_mags_dwarves_45)
    abs_mags_dwarves_58 = dwarf_abs_mag(distance_moduli, app_mags_dwarves_58)
    abs_mags_dwarves_80 = dwarf_abs_mag(distance_moduli, app_mags_dwarves_80)



    abs_mags_dwarfs = {'J': abs_mags_dwarves_J, 'H': abs_mags_dwarves_H, 'K': abs_mags_dwarves_K, '3.6': abs_mags_dwarves_36, \
                   '4.5': abs_mags_dwarves_45, '5.8': abs_mags_dwarves_58, '8.0': abs_mags_dwarves_80}
    
    return dwarf_errs, SpT, abs_mags_dwarfs

#Errors on absolute magnitudes are taken as the errors on apparent magnitudes. 
#The distance error makes a negligible contribution.



#This cell computes the absolute magnitudes and colours for blackbodies of radius R=0.9Rj and R=1.8Rj

c = 299792458 
h = 6.62607004E-34 
k = 1.38064852E-23

constant1 = 2*h*(c**2)
constant2 = (h*c)/k

Tvega = [10800] #Effective temperature of Vega in K
Rvega = 1.643243E9 #Radius of Vega in m
rvega = 7.68 #Distance to Vega in pc
Rjupiter = 69.911E6 #Jupiter radius in m
rbb = 10 #Distance at which we position the bb in pc
constant3 = (Rvega/rvega)**2



#The central wavelengths for the bands we will use (in m)
J = 1.2350005E-6
H = 1.662000E-6
K = 2.15900E-6
ch1 = 3.557259E-6
ch2 = 4.504928E-6
ch3 = 5.738568E-6
ch4 = 7.92734E-6
W_band = 1.4E-6
NB1 = 1.18644E-6
NB2 = 2.09546E-6
z = 0.89615E-6


bbbands = {'J': J, 'H': H, 'K': K, '3.6': ch1, '4.5': ch2, '5.8': ch3, '8.0': ch4, 'W': W_band, \
           'NB1190': NB1, 'NB2090': NB2, 'z': z,}


#We will find the magnitudes at a selection of temperatures. The numbers below can be changed to suit the needs of the user. 



#Defines a function to calculate the spectral energy density at the central wavelength
def planck (T, band):
    fluxes = [(constant1/(band**5))*(1/(np.e**(constant2/(band*i))-1)) for i in T ]
    return fluxes



def bbmags (band, radius, Tbb):
    fluxvega = planck (Tvega, band)
    fluxbb = planck(Tbb, band)
    flux_ratios = [(i/fluxvega[0])*((((radius*Rjupiter)/rbb)**2)/constant3) for i in fluxbb]
    mags = [-2.5*np.log10(i) for i in flux_ratios]
    mags = np.array(mags)
    return mags


J_c = [-9.67994,	 8.16362, -1.33053,	1.11715E-01, -4.82973E-03, 1.00820E-04,	-7.84614E-07]
H_c = [-11.7526, 9.00279	, -1.5037, 0.129202, -0.00580847, 0.000129363, -1.11499E-06]
K_c = [11.0114, 	-0.867471, 0.134163, -0.00642118	, 0.000106693, 0, 0]


c_36 = [9.3422, -3.3522E-1, 6.91081E-2, -3.60108E-3, 6.50191E-5]
c_45 = [9.73946, -4.39968E-1, 7.65343E-2, -3.63435E-3, 	5.82107E-5]
c_58 = [1.10834E1, -9.01820E-1, 1.29019E-1, -6.22795E-3, 1.03507E-4]
c_80 = [9.97853	, -5.29595E-1, 8.43465E-2, -4.12294E-3, 	6.89733E-5]


poly_coeffs = {'J': J_c, 'H': H_c, 'K': K_c, '3.6': c_36, \
       '4.5': c_45, '5.8': c_58, '8.0': c_80}

def poly(band):
    poly_spec = np.arange (6.0, 29.5, 0.5)
    coeffs = np.array(poly_coeffs[band])
    vals = []
    for each in poly_spec:
        y = np.zeros(len(coeffs))
        i = 0
        while i < len(coeffs):
            y[i] = coeffs[i] * (each**i)
            i+=1
        ysum = np.sum(y)
        vals.append(ysum)
    vals = np.array(vals)
    return vals



newcmap = cm.get_cmap('hot_r', 512)
newcmp = ListedColormap(newcmap(np.linspace(0.35, 1.0, 256)))
plt.register_cmap(cmap=newcmp)

newcmap1 = cm.get_cmap('hot_r', 512)
newcmp1 = ListedColormap(newcmap1(np.linspace(0.3, 1.0, 256)))
plt.register_cmap(cmap=newcmp1)

newcmap2 = cm.get_cmap('Greens', 512)
CO_cmap = ListedColormap(newcmap2(np.linspace(0.35, 1.0, 256)))
plt.register_cmap(cmap=CO_cmap)

newcmap3 = cm.get_cmap('Reds_r', 512)
Teff_cmap = ListedColormap(newcmap3(np.linspace(0, 0.65, 256)))
plt.register_cmap(cmap=Teff_cmap)


def add (ax, col, mag, name, col_err = 0, mag_err = 0, c = '#87F9F5'):
    ax.scatter (col, mag, s = 85, c = c, label = name, edgecolor = 'black', zorder = 50)
    ax.errorbar (col, mag, xerr = col_err, yerr = mag_err, c =c, fmt = 'none', zorder = 49)
    ax.legend (fontsize = 'large')
    return ax


def CMD_1 (pfile, bdfile, ax, colour, magnitude, adjusted = True, polynomial = True, bb09 = False, bb18 = False, 
           bbmin = 1000, bbmax = 4000, bbinc = 1000, colourbar = False):
    abs_mags_pl1, adj_pl, pl_errs1, chem, star_names = planet_mags(pfile)
    dwarf_errs, SpT, abs_mags_dwarfs = brown_dwarfs(bdfile)
    
    ax.tick_params(direction = 'in', which = 'major', right = True, top = True, 
                 labelsize = 12, width = 1, length = 9)
    ax.minorticks_on()
    ax.tick_params(direction = 'in', which = 'minor', right = True, top = True, 
                 labelsize = 12, width = 1, length = 4)
    Tbb = np.arange(bbmin, bbmax, bbinc)
    split = colour.split('-')
    col = np.array(abs_mags_dwarfs[split[0]] - abs_mags_dwarfs[split[1]])
    mag = abs_mags_dwarfs[magnitude]
    colour_values = SpT[magnitude]
    im = ax.scatter(col, mag,s=0, c = colour_values, cmap = newcmp, norm=mpl.colors.Normalize(vmin=1.0, vmax=4.0), marker = 'D')
    if colourbar == True:
        cbar = plt.colorbar(im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'Spectral Class')
        cbar.set_ticks([1,1.5, 2, 2.5, 3, 3.5, 4])
        cbar.set_ticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])
    dwarf_yerr = dwarf_errs[magnitude]
    dwarf_xerr = np.array(np.sqrt((dwarf_errs[split[0]])**2 + (dwarf_errs[split[1]])**2))
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=1, vmax=4, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=newcmp)
    diff_color = np.array([(mapper.to_rgba(v)) for v in colour_values])
    for xi, yi, xe, ye, color, cv in zip(col, mag, dwarf_xerr, dwarf_yerr, diff_color, colour_values):
        if cv>4:
            ax.plot(xi, yi, c = 'lightgrey', marker = 'D', markersize=7, zorder = 1)
            ax.errorbar(xi, yi, yerr=ye, xerr=xe, lw=0.7, capsize=0, color='lightgrey', zorder = 1)
        else:
            ax.plot(xi, yi, c = color, marker = 'D', markersize=7, zorder = 1)
            ax.errorbar(xi, yi, yerr=ye, xerr=xe, lw=0.7, capsize=0, color=color, zorder = 1)
        
    newcmp.set_over('lightgrey')
    if '*' in colour and '*' in magnitude:
        xlab = colour.split('*')
        labx = xlab[0]+xlab[1]
        ax.set_xlabel (labx + ' (mag)', fontsize = 'x-large')
        ylab = magnitude.split('*')
        laby = ylab[0]
        ax.set_ylabel (r'$M_{'+laby+'}(mag)$', fontsize = 'x-large')    
    else:
        ax.set_xlabel (colour + ' (mag)', fontsize = 'x-large')
        ax.set_ylabel (r'$M_{'+magnitude+'}(mag)$', fontsize = 'x-large')
    mag_pl = abs_mags_pl1[magnitude]
    mag_pl_adj = adj_pl[magnitude]
    col_pl = np.array(abs_mags_pl1[split[0]] - abs_mags_pl1[split[1]])
    pl_yerr = pl_errs1[magnitude]
    pl_xerr = np.array(np.sqrt((pl_errs1[split[0]])**2 + (pl_errs1[split[0]])**2))
    if adjusted == True:
        ax.scatter (col_pl, mag_pl_adj, c ='#36D6EC', marker = 'o', s=90, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl_adj, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#36D6EC', zorder = 10)
    else:
        ax.scatter (col_pl, mag_pl, c ='#4AB3E7', s=90, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#4AB3E7', zorder = 10)
    if bb09 == True:
        bbmag09 = bbmags(bbbands[magnitude], 0.9, Tbb)
        bbcolour09 = np.array(bbmags(bbbands[split[0]], 0.9, Tbb) - bbmags(bbbands[split[1]], 0.9, Tbb))
        ax.scatter(bbcolour09, bbmag09, marker = 'd', color = 'white', edgecolor = 'black', zorder = 10)
        ax.plot(bbcolour09, bbmag09, color = 'black', zorder = 9)
    if bb18 == True:
        bbmag18 = bbmags(bbbands[magnitude], 1.8, Tbb)
        bbcolour18 = np.array(bbmags(bbbands[split[0]], 1.8, Tbb) - bbmags(bbbands[split[1]], 1.8, Tbb))
        ax.scatter(bbcolour18, bbmag18, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour18, bbmag18, color = 'grey', zorder = 9)
    if polynomial == True:
        ax.plot(np.array(poly(split[0])-poly(split[1])), poly(magnitude), c = '#F9DCFC', linewidth = 9, zorder = 0)    
    ax.invert_yaxis()
    return ax


def CMD_2 (pfile, bdfile, ax, colour, magnitude, adjusted = True, bb09 = True, bb18 = False, bb04 = False, chemical_species = 0, bbmin = 1000, bbmax = 4000, bbinc = 1000,
           colourbar = True):
    abs_mags_pl1, adj_pl, pl_errs1, chem, star_names = planet_mags(pfile)
    dwarf_errs, SpT, abs_mags_dwarfs = brown_dwarfs(bdfile)
    
    ax.tick_params(direction = 'in', which = 'major', right = True, top = True, 
                 labelsize = 12, width = 1, length = 9)
    ax.minorticks_on()
    ax.tick_params(direction = 'in', which = 'minor', right = True, top = True, 
                 labelsize = 12, width = 1, length = 4)
    Tbb = np.arange(bbmin, bbmax, bbinc)
    poly_spec = np.arange (6.0, 29.5, 0.5)
    split = colour.split('-')
    col = np.array(abs_mags_dwarfs[split[0]] - abs_mags_dwarfs[split[1]])
    mag = abs_mags_dwarfs[magnitude]
    dwarf_yerr = dwarf_errs[magnitude]
    dwarf_xerr = np.array(np.sqrt((dwarf_errs[split[0]])**2 + (dwarf_errs[split[1]])**2))
    ax.scatter(col, mag, c = 'lightgrey', marker = 'D', s = 50)
    ax.errorbar(col, mag, xerr=dwarf_xerr, fmt = 'none', yerr=dwarf_yerr, c = 'lightgrey', elinewidth=0.9, barsabove=False, zorder = 0)
    points = np.array([np.array(poly(split[0])-poly(split[1])), poly(magnitude)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=6, vmax=29)
    lc = LineCollection(segments, cmap=newcmp, norm=norm)
    lc.set_array(poly_spec)
    lc.set_linewidth(8)
    im = ax.add_collection(lc)
    if colourbar == True:
        
        cbar = plt.colorbar(im, orientation = 'horizontal', cmap = newcmp, fraction = 0.08, pad = 0.09, label = 'Spectral Class')
        inc = 23/6
        cbar.set_ticks([6,(6+inc), (6+2*inc), (6+3*inc), (6+4*inc), (6+5*inc), 29])
        cbar.set_ticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])
        im.set_clim(6, 29)
    if '*' in colour and '*' in magnitude:
        xlab = colour.split('*')
        labx = xlab[0]+xlab[1]
        ax.set_xlabel (labx + ' (mag)', fontsize = 'x-large')
        ylab = magnitude.split('*')
        laby = ylab[0]
        ax.set_ylabel (r'$M_{'+laby+'}(mag)$', fontsize = 'x-large')  
    else:
        ax.set_xlabel (colour + ' (mag)', fontsize = 'x-large')
        ax.set_ylabel (r'$M_{'+magnitude+'}(mag)$', fontsize = 'x-large')
    mag_pl = abs_mags_pl1[magnitude]
    mag_pl_adj = adj_pl[magnitude]
    col_pl = np.array(abs_mags_pl1[split[0]] - abs_mags_pl1[split[1]])
    ax.invert_yaxis()
    pl_yerr = pl_errs1[magnitude]
    pl_xerr = np.array(np.sqrt((pl_errs1[split[0]])**2 + (pl_errs1[split[0]])**2))
    if adjusted == True:
        ax.scatter (col_pl, mag_pl_adj, c ='#36D6EC', marker = 'o', s=70, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl_adj, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#36D6EC', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl_adj[species], c ='black', marker = '.', s=50, edgecolor = 'black', zorder = 11)
    else:
        ax.scatter (col_pl, mag_pl, c ='#4AB3E7', s=70, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#4AB3E7', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl[species], c ='black', marker = '.', s=50, edgecolor = 'black', zorder = 11)
    if bb04 == True:
        bbmag04 = bbmags(bbbands[magnitude], 0.5, Tbb)
        bbcolour04 = np.array(bbmags(bbbands[split[0]], 0.5, Tbb) - bbmags(bbbands[split[1]], 0.5, Tbb))
        ax.scatter(bbcolour04, bbmag04, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour04, bbmag04, color = 'grey', zorder = 9)
    if bb09 == True:
        bbmag09 = bbmags(bbbands[magnitude], 0.9, Tbb)
        bbcolour09 = np.array(bbmags(bbbands[split[0]], 0.9, Tbb) - bbmags(bbbands[split[1]], 0.9, Tbb))
        ax.scatter(bbcolour09, bbmag09, marker = 'd', color = 'white', edgecolor = 'black', zorder = 10)
        ax.plot(bbcolour09, bbmag09, color = 'black', zorder = 9)
    if bb18 == True:
        bbmag18 = bbmags(bbbands[magnitude], 1.8, Tbb)
        bbcolour18 = np.array(bbmags(bbbands[split[0]], 1.8, Tbb) - bbmags(bbbands[split[1]], 1.8, Tbb))
        ax.scatter(bbcolour18, bbmag18, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour18, bbmag18, color = 'grey', zorder = 9)
    return ax


def CMD_3 (pfile, bdfile, ax, colour, magnitude, adjusted = True, bb09 = True, bb18 = False, bb04 = False, chemical_species = 0, 
           bbmin = 1000, bbmax = 4000, bbinc = 1000, highlight = 0, colourbar = True):
    abs_mags_pl1, adj_pl, pl_errs1, chem, star_names = planet_mags(pfile)
    dwarf_errs, SpT, abs_mags_dwarfs = brown_dwarfs(bdfile)
    
    ax.tick_params(direction = 'in', which = 'major', right = True, top = True, 
                 labelsize = 12, width = 1, length = 9)
    ax.minorticks_on()
    ax.tick_params(direction = 'in', which = 'minor', right = True, top = True, 
                 labelsize = 12, width = 1, length = 4)
    Tbb = np.arange(bbmin, bbmax, bbinc)
    poly_spec = np.arange (6.0, 29.5, 0.5)
    split = colour.split('-')
    col = np.array(abs_mags_dwarfs[split[0]] - abs_mags_dwarfs[split[1]])
    mag = abs_mags_dwarfs[magnitude]
    dwarf_yerr = dwarf_errs[magnitude]
    dwarf_xerr = np.array(np.sqrt((dwarf_errs[split[0]])**2 + (dwarf_errs[split[1]])**2))
    ax.scatter(col, mag, c = 'lightgrey', marker = 'D', s = 50)
    ax.errorbar(col, mag, xerr=dwarf_xerr, fmt = 'none', yerr=dwarf_yerr, c = 'lightgrey', elinewidth=0.9, barsabove=False, zorder = 0)
    points = np.array([np.array(poly(split[0])-poly(split[1])), poly(magnitude)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mpl.colors.Normalize(vmin=1, vmax=35)
    lc = LineCollection(segments, cmap=newcmp, norm=norm)
    lc.set_array(poly_spec)
    lc.set_linewidth(8)
    im = ax.add_collection(lc)
    if colourbar == True:
        
        cbar = plt.colorbar(im, orientation = 'horizontal', cmap = newcmp, fraction = 0.08, pad = 0.09, label = 'Spectral Class')
        inc = 23/6
        cbar.set_ticks([6,(6+inc), (6+2*inc), (6+3*inc), (6+4*inc), (6+5*inc), 29])
        cbar.set_ticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])
        im.set_clim(6, 29)
    ax.set_xlabel (colour + ' (mag)', fontsize = 'x-large')
    ax.set_ylabel (r'$M_{'+magnitude+'}(mag)$', fontsize = 'x-large')
    mag_pl = abs_mags_pl1[magnitude]
    mag_pl_adj = adj_pl[magnitude]
    col_pl = np.array(abs_mags_pl1[split[0]] - abs_mags_pl1[split[1]])
    ax.invert_yaxis()
    pl_yerr = pl_errs1[magnitude]
    pl_xerr = np.array(np.sqrt((pl_errs1[split[0]])**2 + (pl_errs1[split[0]])**2))
    if adjusted == True:
        ax.scatter (col_pl, mag_pl_adj, c ='grey', marker = 'o', s=70, zorder = 11)
        ax.scatter (col_pl, mag_pl_adj, c ='lightgrey', marker = 'o', s=35, zorder = 12)
        ax.errorbar(col_pl, mag_pl_adj, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=0.8, barsabove=False, color = 'grey', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl_adj[species], c ='black', marker = '.', s=50, edgecolor = 'black', zorder = 11)
    else:
        ax.scatter (col_pl, mag_pl, c ='grey', s=45, zorder = 11)
        ax.scatter (col_pl, mag_pl, c ='lightgrey', s=25, zorder = 12)
        ax.errorbar(col_pl, mag_pl, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=0.8, barsabove=False, color = 'grey', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl[species], c ='black', marker = '.', s=50, edgecolor = 'black', zorder = 11)
    if highlight != 0:
        highlight = np.array(highlight)
        if np.shape(highlight) == (2,):
            ind = np.argwhere (star_names == highlight[0])
            ax.scatter (col_pl[ind], mag_pl_adj[ind], c = 'white', s = 120, marker = 'o', zorder = 14)
            ax.scatter (col_pl[ind], mag_pl_adj[ind], c = highlight[1], s = 140, marker = r'$\bigodot$', zorder = 15, label = highlight[0]+'b')
            ax.errorbar(col_pl[ind], mag_pl_adj[ind], fmt='none', xerr=float(pl_xerr[ind]), \
                 yerr=float(pl_yerr[ind]), elinewidth=0.8, barsabove=False, color = highlight[1], zorder = 13)
        else:
            for a in highlight:
                ind = np.argwhere (star_names == a[0])
                cs = a[1]
                for i in ind:
                    ax.scatter (col_pl[i], mag_pl_adj[i], c = 'white', s = 120, marker = 'o', zorder = 14)
                    ax.scatter (col_pl[i], mag_pl_adj[i], c = cs, s = 140, marker = r'$\bigodot$', zorder = 15, label = a[0]+'b')
                    ax.errorbar(col_pl[i], mag_pl_adj[i], fmt='none', xerr=pl_xerr[i], \
                    yerr=pl_yerr[i], elinewidth=0.8, barsabove=False, color = cs, zorder = 13)
    if bb04 == True:
        bbmag04 = bbmags(bbbands[magnitude], 0.5, Tbb)
        bbcolour04 = np.array(bbmags(bbbands[split[0]], 0.5, Tbb) - bbmags(bbbands[split[1]], 0.5, Tbb))
        ax.scatter(bbcolour04, bbmag04, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour04, bbmag04, color = 'grey', zorder = 9)
    if bb09 == True:
        bbmag09 = bbmags(bbbands[magnitude], 0.9, Tbb)
        bbcolour09 = np.array(bbmags(bbbands[split[0]], 0.9, Tbb) - bbmags(bbbands[split[1]], 0.9, Tbb))
        ax.scatter(bbcolour09, bbmag09, marker = 'd', color = 'white', edgecolor = 'black', zorder = 10)
        ax.plot(bbcolour09, bbmag09, color = 'black', zorder = 9)
    if bb18 == True:
        bbmag18 = bbmags(bbbands[magnitude], 1.8, Tbb)
        bbcolour18 = np.array(bbmags(bbbands[split[0]], 1.8, Tbb) - bbmags(bbbands[split[1]], 1.8, Tbb))
        ax.scatter(bbcolour18, bbmag18, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour18, bbmag18, color = 'grey', zorder = 9)
    ax.legend(fontsize = 'large')
    return ax

    

def synth_SpT_complete(spt):
    complete = []
    for i in spt:
        if len(i) < 4:
            new = i +'.0'
            complete.append(new)
        else:
            complete.append(i)
    complete = np.array (complete)
    return(complete)

def synth_import(fname):
    synthetic_data = np.genfromtxt(fname, comments= '#', unpack = True)
    synthetic_data_str = np.genfromtxt(fname, comments= '#', unpack = True, dtype = str)
    synth_SpT = synthetic_data_str[1]
    synth_SpT = synth_SpT_complete(synth_SpT)
    synth_SpT_num = spectral_class (synth_SpT)
    synth_distance = synthetic_data [18]
    synth_J = synthetic_data [2]
    synth_J_err = synthetic_data [3]
    synth_H = synthetic_data [4]
    synth_H_err = synthetic_data [5]
    synth_K = synthetic_data [6]
    synth_K_err = synthetic_data [7]
    synth_W = synthetic_data [8]
    synth_W_err = synthetic_data [9]
    synth_NB1 = synthetic_data [10]
    synth_NB1_err = synthetic_data [11]
    synth_NB2 = synthetic_data [12]
    synth_NB2_err = synthetic_data [13]
    synth_z = synthetic_data [14]
    synth_z_err = synthetic_data [15]
    synth_new = synthetic_data [16]
    synth_new_err = synthetic_data [17]
    synth_J_abs = np.array(synth_J - (5*np.log10(synth_distance)) + 5)
    synth_H_abs = np.array(synth_H - (5*np.log10(synth_distance)) + 5)
    synth_K_abs = np.array(synth_K - (5*np.log10(synth_distance)) + 5)
    synth_W_abs = np.array(synth_W - (5*np.log10(synth_distance)) + 5)
    synth_NB1_abs = np.array(synth_NB1 - (5*np.log10(synth_distance)) + 5)
    synth_NB2_abs = np.array(synth_NB2 - (5*np.log10(synth_distance)) + 5)
    synth_z_abs = np.array(synth_z - (5*np.log10(synth_distance)) + 5)
    synth_new_abs = np.array(synth_new - (5*np.log10(synth_distance)) + 5)

    synth_abs = {'J': synth_J_abs, 'H': synth_H_abs, 'K': synth_K_abs, 'W': synth_W_abs, 
                 'NB1190': synth_NB1_abs, 'NB2090': synth_NB2_abs, 'z': synth_z_abs, 'new': synth_new_abs}
    synth_errs = {'J': synth_J_err, 'H': synth_H_err, 'K': synth_K_err, 'W': synth_W_err, 
                  'NB1190': synth_NB1_err, 'NB2090': synth_NB2_err, 'z': synth_z_err, 'new': synth_new_err}
    
    return synth_abs, synth_errs, synth_SpT_num




def colorbar(size):
    import pylab as pl
    a = np.array([[0,1]])
    plt.figure(figsize=(size, 0.8))
    img = plt.imshow(a, cmap=newcmp)
    plt.gca().set_visible(False)
    cax = plt.axes([0.05, 0.65, 0.9, 0.3])
    cbar = plt.colorbar(orientation="horizontal", fraction = 0.08, cax=cax, label = 'Spectral Class')
    inc = 1/6
    cbar.set_ticks([0,(inc), (2*inc), (3*inc), (4*inc), (5*inc), 1])
    cbar.set_label(label='Spectral Class',size=20)
    cbar.set_ticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])
    cbar.ax.tick_params(labelsize=17)
    return cbar
    

def CMD_synth (pfile, ax, colour, magnitude, adjusted = True, polynomial = True, bb09 = True, bb18 = True, chemical_species = 0,
               synth_file = 'synth_mags.txt', bbmin = 1000, bbmax = 4000, bbinc = 1000,
               colourbar = True):
    abs_mags_pl1, adj_pl, pl_errs1, chem, star_names = planet_mags(pfile)
    synth_abs, synth_errs, synth_SpT_num = synth_import(synth_file)
    Tbb = np.arange(bbmin, bbmax, bbinc)
    
    ax.tick_params(direction = 'in', which = 'major', right = True, top = True, 
                 labelsize = 12, width = 1, length = 9)
    ax.minorticks_on()
    ax.tick_params(direction = 'in', which = 'minor', right = True, top = True, 
                 labelsize = 12, width = 1, length = 4)
    split = colour.split('-')
    col = np.array(synth_abs[split[0]] - synth_abs[split[1]])
    mag = synth_abs[magnitude]
    colour_values = synth_SpT_num
    im = ax.scatter(col, mag,s=0, c = colour_values, cmap = newcmp, norm=mpl.colors.Normalize(vmin=1.0, vmax=4.0), marker = 'D')
    if colourbar ==True:    
        cbar = plt.colorbar(im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'Spectral Class')
        cbar.set_ticks([1,1.5, 2, 2.5, 3, 3.5, 4])
        cbar.set_ticklabels(['M0', 'M5', 'L0', 'L5', 'T0', 'T5', 'Y0'])
    dwarf_yerr = synth_errs[magnitude]
    dwarf_xerr = np.array(np.sqrt((synth_errs[split[0]])**2 + (synth_errs[split[1]])**2))
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=1, vmax=4, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=newcmp)
    diff_color = np.array([(mapper.to_rgba(v)) for v in colour_values])
    for xi, yi, xe, ye, color, cv in zip(col, mag, dwarf_xerr, dwarf_yerr, diff_color, colour_values):
        if cv>4:
            ax.plot(xi, yi, c = 'lightgrey', marker = 'D', markersize=6, zorder = 1)
            ax.errorbar(xi, yi, yerr=ye, xerr=xe, lw=0.7, capsize=0, color='lightgrey', zorder = 1)
        else:
            ax.plot(xi, yi, c = color, marker = 'D', markersize=6, zorder = 1)
            ax.errorbar(xi, yi, yerr=ye, xerr=xe, lw=0.7, capsize=0, color=color, zorder = 1)
        
    newcmp.set_over('lightgrey')
    ax.invert_yaxis()
    if '*' in colour and '*' in magnitude:
        xlab = colour.split('*')
        labx = xlab[0]+xlab[1]
        ax.set_xlabel (labx + ' (mag)', fontsize = 'x-large')
        ylab = magnitude.split('*')
        laby = ylab[0]
        ax.set_ylabel (r'$M_{'+laby+'}(mag)$', fontsize = 'x-large')    
    else:
        ax.set_xlabel (colour + ' (mag)', fontsize = 'x-large')
        ax.set_ylabel (r'$M_{'+magnitude+'}(mag)$', fontsize = 'x-large')
    mag_pl = abs_mags_pl1[magnitude]
    mag_pl_adj = adj_pl[magnitude]
    col_pl = np.array(abs_mags_pl1[split[0]] - abs_mags_pl1[split[1]])
    pl_yerr = pl_errs1[magnitude]
    pl_xerr = np.array(np.sqrt((pl_errs1[split[0]])**2 + (pl_errs1[split[0]])**2))
    if adjusted == True:
        ax.scatter (col_pl, mag_pl_adj, c ='#36D6EC', marker = 'o', s=70, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl_adj, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#36D6EC', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl_adj[species], c ='black', marker = '.', s=50, edgecolor = 'black', zorder = 11)
    else:
        ax.scatter (col_pl, mag_pl, c ='#4AB3E7', s=70, edgecolor = 'black', zorder = 11)
        ax.errorbar(col_pl, mag_pl, fmt='none', xerr=pl_xerr, \
                 yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#4AB3E7', zorder = 10)
        if chemical_species!=0:
            species = np.argwhere(chem[chemical_species]==1)
            ax.scatter (col_pl[species], mag_pl[species], c ='black', marker = '.', s=50, zorder = 11)
    if bb09 == True:
        bbmag09 = bbmags(bbbands[magnitude], 0.9, Tbb)
        bbcolour09 = np.array(bbmags(bbbands[split[0]], 0.9, Tbb) - bbmags(bbbands[split[1]], 0.9, Tbb))
        ax.scatter(bbcolour09, bbmag09, marker = 'd', color = 'white', edgecolor = 'black', zorder = 10)
        ax.plot(bbcolour09, bbmag09, color = 'black', zorder = 9)
    if bb18 == True:
        bbmag18 = bbmags(bbbands[magnitude], 1.8, Tbb)
        bbcolour18 = np.array(bbmags(bbbands[split[0]], 1.8, Tbb) - bbmags(bbbands[split[1]], 1.8, Tbb))
        ax.scatter(bbcolour18, bbmag18, marker = 'd', color = 'white', edgecolor = 'grey', zorder = 10)
        ax.plot(bbcolour18, bbmag18, color = 'grey', zorder = 9)
    #if polynomial == True:
    #    ax.plot(np.array(poly(split[0])-poly(split[1])), poly(magnitude), c = '#F9DCFC', linewidth = 9, zorder = 0)    
    return ax




SpT_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', 
                                             [(0,    '#ae0001'),
                                               (0.33, '#ae0001'),
                                                (0.58, '#ae0001'),
                                                 (0.59, '#000000'),
                                                  (0.60, '#000000'),
                                                 (0.61, '#000000'),
                                                  (0.62, '#000000'),
                                                   (0.63, '#000000'),
                                                    (0.64, '#000000'),
                                              (0.65, '#000000'),
                                               (0.66, '#eeba30'),
                                              (1,    '#eeba30')], N=126)
                                               
SpTT_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', 
                                             [(0,    '#ECC036'),
                                               (0.25, '#EC9736'),
                                              (0.5, '#EC6A36'),
                                               (0.75, '#EC3E36'),
                                              (1,    '#EC3E36')], N=126)
                                               
                                               
                                               
FeH_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', 
                                             [(0,    '#efbbff'),
                                               (0.25, '#d896ff'),
                                              (0.5, '#be29ec'),
                                               (0.75, '#800080'),
                                              (1,    '#660066')], N=126)
                                               
logg_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', 
                                             [(0,    '#a67c00'),
                                               (0.25, '#bf9b30'),
                                              (0.5, '#ffbf00'),
                                               (0.75, '#ffcf40'),
                                              (1,    '#ffdc73')], N=126)                                              
                                               
topc = cm.get_cmap('Oranges_r', 128)
bottomc = cm.get_cmap('Blues', 128)

newcolors = np.vstack((topc(np.linspace(0, 1, 128)),
                       bottomc(np.linspace(0, 1, 128))))
newcmp_CO = ListedColormap(newcolors, name='OrangeBlue')




def CMD_model (pfile, bdfile, ax, colour, magnitude, SpT=50, FeH=50, logg=50, CO=50, Teff=50, planets = True, 
               add_brown_dwarfs = True, adjusted = True, 
               synth_file = 'synth_mags.txt', 
               bbmin = 1000, bbmax = 4000, bbinc = 1000, colour_by = 'C/O Ratio'):
    abs_mags_pl1, adj_pl, pl_errs1, chem, star_names = planet_mags(pfile)
    dwarf_errs, SpT_bd, abs_mags_dwarfs = brown_dwarfs(bdfile)
    
    Tbb = np.arange(bbmin, bbmax, bbinc)
    synth_abs, synth_errs, synth_SpT_num = synth_import(synth_file)

    ax.tick_params(direction = 'in', which = 'major', right = True, top = True, 
                 labelsize = 12, width = 1, length = 9)
    ax.minorticks_on()
    ax.tick_params(direction = 'in', which = 'minor', right = True, top = True, 
                 labelsize = 12, width = 1, length = 4)
    import models
    #poly_spec = np.arange (6.0, 29.5, 0.5)
    model_mags = models.mags(magnitude, Teff, SpT, FeH, logg, CO, table = False) 
    model_colours = models.colours(colour, Teff, SpT, FeH, logg, CO, table = False)
    
    Teffs = [float(i) for i in (model_mags[:,1])]
    loggs = [float(i) for i in (model_mags[:,2])]
    FeHs = [float(i) for i in (model_mags[:,3])]
    COs = [float(i) for i in (model_mags[:,4])]
    SpTs = model_mags[:,5]
    mags = [float(i) for i in (model_mags[:,6])]

    
    cols = [float(i) for i in (model_colours[:,6])]
    

    spt = np.arange(len(cols))
    
    if colour_by == 'Spectral Type':
        im = ax.scatter (cols, mags, s = 0, c = spt, norm=mpl.colors.Normalize(vmin=0, vmax=len(cols)), cmap = SpTT_cmap)
        cbar = plt.colorbar (im, orientation = 'horizontal', cmap = SpTT_cmap, fraction = 0.08, pad = 0.09, 
                             label = 'Spectral Type of Parent Star')
        cbar.set_ticks([0, (len(cols)/3), (2*len(cols)/3), len(cols)])
        cbar.set_ticklabels(['F5', 'G5', 'K5', 'M5'])
        for i, j, k in zip (mags, SpTs, cols):
            if j == 'F5':
                ax. scatter (k, i, s = 45, c = '#ECC036')
            elif j == 'G5':
                ax. scatter (k, i, s = 45, c = '#EC9736')
            elif j == 'K5':
                ax. scatter (k, i, s = 45, c = '#EC6A36')
            elif j == 'M5':
                ax. scatter (k, i, s = 45, c = '#EC3E36')
    
    elif colour_by == 'Metallicity':
        im = ax.scatter(cols, mags, s = 45, c = FeHs, cmap = FeH_cmap)
        plt.colorbar (im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'Metallicity')
        
    elif colour_by == 'Surface Gravity':
        im = ax.scatter(cols, mags, s = 45, c = loggs, cmap = logg_cmap)
        plt.colorbar (im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'Planetary log(g)')
        
    elif colour_by == 'C/O Ratio':
        im = ax.scatter(cols, mags, s = 90, c = COs, cmap = SpT_cmap)
        plt.colorbar (im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'C/O Ratio')
        
    elif colour_by == 'Effective Temperature':
        im = ax.scatter(cols, mags, s = 45, c = Teffs, cmap = Teff_cmap)
        plt.colorbar (im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'Effective Temperature')
        
    else:
        im = ax.scatter(cols, mags, s = 45, c = COs, cmap = SpT_cmap)
        cbar = plt.colorbar (im, orientation = 'horizontal', fraction = 0.08, pad = 0.09, label = 'C/O Ratio')
        cbar.set_label(label='C/O Ratio' ,size=12)
        
        cbar.ax.tick_params(labelsize=12)
                        
    ax.set_xlabel (colour + ' (mag)', fontsize = 'x-large')
    ax.set_ylabel (r'$M_{'+magnitude+'}(mag)$', fontsize = 'x-large')
    split = colour.split('-') 
    if add_brown_dwarfs == True:
        if magnitude in abs_mags_dwarfs and split[0] in abs_mags_dwarfs and split[1] in abs_mags_dwarfs:
            col = np.array(abs_mags_dwarfs[split[0]] - abs_mags_dwarfs[split[1]])
            mag = abs_mags_dwarfs[magnitude]
        else:
            col = np.array(synth_abs[split[0]] - synth_abs[split[1]])
            mag = synth_abs[magnitude]
        ax.scatter(col, mag, c = 'lightgrey', marker = 'D', s = 50, zorder = 0)
    else:
        col, mag = cols, mags
    if planets == True:
        if magnitude in abs_mags_pl1 and split[0] in abs_mags_pl1 and split[1] in abs_mags_pl1:
            mag_pl = abs_mags_pl1[magnitude]
            mag_pl_adj = adj_pl[magnitude]
            col_pl = np.array(abs_mags_pl1[split[0]] - abs_mags_pl1[split[1]])
            pl_yerr = pl_errs1[magnitude]
            pl_xerr = np.array(np.sqrt((pl_errs1[split[0]])**2 + (pl_errs1[split[0]])**2))
            if adjusted == True:
                ax.scatter (col_pl, mag_pl_adj, c ='#36D6EC', marker = 'o', s=70, edgecolor = 'black', zorder = 11)
                ax.errorbar(col_pl, mag_pl_adj, fmt='none', xerr=pl_xerr, \
                        yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#36D6EC', zorder = 10)
            else:
                ax.scatter (col_pl, mag_pl, c ='#4AB3E7', s=70, edgecolor = 'black', zorder = 11)
                ax.errorbar(col_pl, mag_pl, fmt='none', xerr=pl_xerr, \
                            yerr=pl_yerr, elinewidth=1, barsabove=False, color = '#4AB3E7', zorder = 10)
        else:
            mag_pl, mag_pl_adj, col_pl, pl_yerr, pl_xerr = mags, mags, cols, np.zeros(len(mags)), np.zeros(len(mags))
    ax.invert_yaxis()
    return ax

