#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:09:45 2020

@author: Georgina Dransfield
"""

import numpy as np
from scipy import interpolate
import os 
from astropy.io import fits
from tabulate import tabulate



loc = os.getcwd()

c = 299792458 

vega_wavelength, vega_flux = np.genfromtxt(os.path.join(loc, 'files', 'vega_data.txt'), comments = '#', unpack = True)


vega_wavelength = np.array(vega_wavelength * 1E-9)
conv_factors = np.array([(i**2)*((1E11)/c)*(4*np.pi)*(1E23) for i in vega_wavelength])


vega_flux = np.array(vega_flux * conv_factors)

Jresponse = os.path.join(loc, 'files', "Jresponse.txt")
Jrf = np.genfromtxt (Jresponse, comments = '#', unpack = True)
Jrf[0] = np.array (Jrf[0]* 1E-6)
Hresponse = os.path.join(loc, 'files', "Hresponse.txt")
Hrf = np.genfromtxt (Hresponse, comments = '#', unpack = True)
Hrf[0] = np.array (Hrf[0]* 1E-6)
Kresponse = os.path.join(loc, 'files', 'Kresponse.txt')
Krf = np.genfromtxt (Kresponse, comments = '#', unpack = True)
Krf[0] = np.array (Krf[0]* 1E-6)

NB1response = os.path.join(loc, 'files', 'hawki_NB1190.dat')
NB1rf = np.genfromtxt (NB1response, comments = '#', unpack = True)
NB1rf[0] = np.array (NB1rf[0]* 1E-9)
NB1rf[1] = np.array(NB1rf[1]/100)

NB2response = os.path.join(loc, 'files', 'hawki_NB2090.dat')
NB2rf = np.genfromtxt (NB2response, comments = '#', unpack = True)
NB2rf[0] = np.array (NB2rf[0]* 1E-9)
NB2rf[1] = np.array(NB2rf[1]/100)

zfile = os.path.join(loc, 'files', 'sloan_z.txt')
zdat = np.genfromtxt (zfile, comments = '#', unpack =True)
zwavs = np.array(zdat[0]*1E-10)
zrf = np.array (zdat[1]/np.max(zdat[1]))

IRAC1_rf = np.genfromtxt(os.path.join(loc, 'files', '080924ch1trans_full.txt'), comments = '#', unpack = True)
IRAC1_rf[0] = np.array(IRAC1_rf[0]*1E-6)
IRAC1_rf[1] = np.array([i/np.max(IRAC1_rf) for i in IRAC1_rf[1]])

IRAC2_rf = np.genfromtxt(os.path.join(loc, 'files', '080924ch2trans_full.txt'), comments = '#', unpack = True)
IRAC2_rf[0] = np.array(IRAC2_rf[0]*1E-6)
IRAC2_rf[1] = np.array([i/np.max(IRAC2_rf) for i in IRAC2_rf[1]])

IRAC3_rf = np.genfromtxt(os.path.join(loc, 'files', '080924ch3trans_full.txt'), comments = '#', unpack = True)
IRAC3_rf[0] = np.array(IRAC3_rf[0]*1E-6)
IRAC3_rf[1] = np.array([i/np.max(IRAC3_rf) for i in IRAC3_rf[1]])

IRAC4_rf = np.genfromtxt(os.path.join(loc, 'files', '080924ch4trans_full.txt'), comments = '#', unpack = True)
IRAC4_rf[0] = np.array(IRAC4_rf[0]*1E-6)
IRAC4_rf[1] = np.array([i/np.max(IRAC4_rf) for i in IRAC4_rf[1]])

#G141 Response function imported
hdulist = fits.open (os.path.join(loc, 'files', 'wfc3_ir_g141_src_004_syn.fits'))
data1 = hdulist[1].data



#Function to split the data into indivudual arrays
def get_stuff (datum, index):
    wav = []
    for each in datum:
        x = each[index]
        wav.append(x)
    return wav

wavs = np.array(get_stuff(data1, 0))
wavs = np.array(wavs*1E-10) #converting wavelengths from Angstrom to meters
rf_data = get_stuff(data1, 1)
rf_data = np.array(rf_data/np.max(rf_data)) #normalising the response function to 1

Wwavmin = int(np.argwhere (wavs > 1.3E-6)[0]) #cutting the grism for a synthetic water band centered on 1.410 microns
Wwavmax = int(np.argwhere (wavs < 1.52E-6)[-1])

Wwavs = np.array (wavs[Wwavmin:Wwavmax])
Wrf = np.array (rf_data[Wwavmin:Wwavmax])



def CO_from_files (file_names): 
    CO = []                     
    for each in file_names:
        COratio = float(each[-8: -4])
        CO.append(COratio)
    return CO


def get_data (files, CO_ratios):  
    i=0                
    mega_data = {}     
    for each in files:
        data = np.genfromtxt(files[i], comments = '#', unpack = True)
        mega_data[CO_ratios[i]] = data
        i+=1
    return mega_data
    

def extract_wavelength (files): 
    data = []
    for each in files:
        new_data = files[each][0]
        data.append(new_data)
    return data

def extract_flux (files): 
    data = []
    for each in files:
        new_data = files[each][1]
        data.append(new_data)
    return data



def cut_data (wavelength, flux1, lower_limit, upper_limit):
    new_wavelength = []
    new_flux = []
    i=0
    while i<len(wavelength):
        index1 = np.nonzero(wavelength[i]>lower_limit)
        index2 = np.nonzero(wavelength[i]<upper_limit)
        new_wavelength.append (wavelength[i][index1[0][0]:index2[0][-1]])
        new_flux.append (flux1 [i][index1[0][0]:index2[0][-1]])
        i+=1
    return new_wavelength, new_flux


Jmin = 1.105E-6
Jmax = 1.349E-6

Hmin = 1.504E-6
Hmax = 1.709E-6

Kmin = 1.989E-6
Kmax = 2.316E-6

Wmin = 1.325E-6
Wmax = 1.495E-6

IRAC1min = 3.179E-6
IRAC1max = 3.955E-6

IRAC2min = 3.955E-6
IRAC2max = 5.015E-6

IRAC3min = 5.015E-6
IRAC3max = 6.442E-6

IRAC4min = 6.442E-6
IRAC4max = 9.343E-6

NB1min = 1.171716E-6
NB1max = 1.199355E-6

NB2min = 2.076706E-6
NB2max = 2.113253E-6

zmin = 0.796044E-6
zmax = 1.083325E-6



def fluxes1 (fluxx, wavelengths):
    flux_data = []
    i = 0
    while i<len(wavelengths):
        flux1 = np.trapz(fluxx[i], wavelengths[i])
        flux_data.append(flux1)
        i+=1
    output = np.array (flux_data)
    return output


def cut_vega (vega_wav, vega_fl, low, high):
    index_low = int((np.argwhere (vega_wavelength>low))[0])
    index_high = int((np.argwhere (vega_wavelength<high))[-1])
    new_vega_wav = vega_wav [index_low : index_high]
    new_vega_fl = vega_fl [index_low : index_high]
    return new_vega_wav, new_vega_fl

def rf (wav, fl, rfband): #function scales the fluxes by the 
    j = 0                   #response function of each band
    rf_fluxes_all = []
    while j<len(wav):
        rf_fluxes = []
        i=0
        while i<len(wav[j]):
            rf_flux = (fl[j][i])*rfband(wav[j][i])
            rf_fluxes.append(rf_flux)
            i+=1
        rff = np.array(rf_fluxes)
        rf_fluxes_all.append(rff)
        j+=1
    rffa = np.array(rf_fluxes_all)
    return rffa

model_map = np.genfromtxt('models_map.txt', 
                          usecols = (0, 1, 2, 3, 4), comments = '#', unpack = True)
model_map_str = np.genfromtxt('models_map.txt', comments = '#', 
                            usecols = (0, 5), unpack = True, dtype = str)

Model_numbers = model_map_str[0]
Teff_vals = model_map[1]
logg_vals = model_map[2]
FeH_vals = model_map[3]
CO_vals = model_map[4]
SpT_vals = model_map_str[1]

Teffs = [1000, 1250, 1500, 1750, 2000, 2250, 2500]
COs = [0.35, 0.55, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.85, 0.90, 0.91, 0.92, 
      0.93, 0.94, 0.95, 1.00, 1.05, 1.12, 1.40]
loggs = [2.3, 3.0, 4.0, 5.0]
FeHs = [-0.5, 0.0, 0.5, 1.0, 2.0]
SpTs = ['F5', 'G5', 'K5', 'M5']

def mags (band, Teff = 50, SpT = 50, FeH = 50, logg = 50, CO = 50, table = True):   
    if Teff != 50:
        if type (Teff) == list:
            Teff_index = [np.argwhere (Teff_vals == i) for i in Teff]
        else:
            Teff_index = np.argwhere (Teff_vals == Teff)
    else:
        Teff_index = np.arange (1, 10641)
        
    if SpT != 50:
        if type(SpT) == list:
            SpT_index = [np.argwhere (SpT_vals == i) for i in SpT] 
        else:
            SpT_index = np.argwhere (SpT_vals == SpT)
    else:
        SpT_index = np.arange (1, 10641)
    
    if FeH != 50:
        if type (FeH) == list:
            FeH_index = [np.argwhere(FeH_vals == i) for i in FeH]
        else:
            FeH_index = np.argwhere (FeH_vals == FeH)
    else:
        FeH_index = np.arange (1, 10641)
        
    if logg != 50:
        if type(logg) == list:
            logg_index = [np.argwhere(logg_vals == i) for i in logg]
        else:
            logg_index = np.argwhere (logg_vals == logg)
    else:
        logg_index = np.arange (1, 10641)
        
    if CO != 50:
        if type(CO) == list:
            CO_index = [np.argwhere(CO_vals == i) for i in CO]
        else:
            CO_index = np.argwhere (CO_vals == CO)
    else:
        CO_index = np.arange (1, 10641)
    
    overlap1 = np.intersect1d(Teff_index, SpT_index)
    overlap2 = np.intersect1d(overlap1, logg_index)
    overlap3 = np.intersect1d(overlap2, FeH_index)
    overlap4 = np.intersect1d(overlap3, CO_index)
    
    global overlap
    overlap = overlap4
    
    Models = [Model_numbers[i] for i in overlap4]
    filenames = [i + '.txt' for i in Models]
    filepaths = [os.path.join(loc, 'model_spectra', i) for i in filenames]
    

    data_dict = get_data (filepaths, Models)
    wavelengths = np.array(extract_wavelength (data_dict))
    wavelengths = wavelengths / 100
    fluxes = np.array(extract_flux (data_dict))
    conv = ((1/1000) * (1E26))
    fluxes = fluxes * conv
    
    
    #Interpolating the response function for each band to create a function
    intJ = interpolate.interp1d (Jrf[0], Jrf[1])
    intH = interpolate.interp1d (Hrf[0], Hrf[1])
    intK = interpolate.interp1d (Krf[0], Krf[1])
    intW = interpolate.interp1d (Wwavs, Wrf)
    intNB2 = interpolate.interp1d (NB2rf[0], NB2rf[1])
    intNB1 = interpolate.interp1d (NB1rf[0], NB1rf[1])
    intz = interpolate.interp1d (zwavs, zrf)
    intIRAC1 = interpolate.interp1d (IRAC1_rf[0], IRAC1_rf[1])
    intIRAC2 = interpolate.interp1d (IRAC2_rf[0], IRAC2_rf[1])
    intIRAC3 = interpolate.interp1d (IRAC3_rf[0], IRAC3_rf[1])
    intIRAC4 = interpolate.interp1d (IRAC4_rf[0], IRAC4_rf[1])
    
    J_wavelength, J_flux = cut_data (wavelengths, fluxes, Jmin, Jmax) 
    H_wavelength, H_flux = cut_data (wavelengths, fluxes, Hmin, Hmax) 
    K_wavelength, K_flux = cut_data (wavelengths, fluxes, Kmin, Kmax)
    W_wavelength, W_flux = cut_data (wavelengths, fluxes, Wmin, Wmax)
    
    
    J_flux = np.array(rf(J_wavelength, J_flux, intJ))
    H_flux = np.array(rf(H_wavelength, H_flux, intH))
    K_flux = np.array(rf(K_wavelength, K_flux, intK))
    W_flux = np.array(rf(W_wavelength, W_flux, intW))
    
    IRAC1_data = cut_data (wavelengths, fluxes, IRAC1min, IRAC1max) 
    IRAC1_wavelength = IRAC1_data [0]
    IRAC1_flux = IRAC1_data [1]
    
    IRAC2_data = cut_data (wavelengths, fluxes, IRAC2min, IRAC2max) 
    IRAC2_wavelength = IRAC2_data [0]
    IRAC2_flux = IRAC2_data [1]
    
    IRAC3_data = cut_data (wavelengths, fluxes, IRAC3min, IRAC3max) 
    IRAC3_wavelength = IRAC3_data [0]
    IRAC3_flux = IRAC3_data [1]
    
    IRAC4_data = cut_data (wavelengths, fluxes, IRAC4min, IRAC4max) 
    IRAC4_wavelength = IRAC4_data [0]
    IRAC4_flux = IRAC4_data [1]
    
    NB2_data = cut_data (wavelengths, fluxes, NB2min, NB2max) 
    NB2_wavelength = NB2_data [0]
    NB2_flux = NB2_data [1]
    
    NB2_flux = np.array(rf(NB2_wavelength, NB2_flux, intNB2))
    
    NB1_data = cut_data (wavelengths, fluxes, NB1min, NB1max) 
    NB1_wavelength = NB1_data [0]
    NB1_flux = NB1_data [1]
    
    NB1_flux = np.array(rf(NB1_wavelength, NB1_flux, intNB1))
    
    z_data = cut_data (wavelengths, fluxes, zmin, zmax) 
    z_wavelength = z_data [0]
    z_flux = z_data [1]
    
    z_flux = np.array(rf(z_wavelength, z_flux, intz))
    
    J_vega_wav, J_vega_fl = cut_vega(vega_wavelength, vega_flux, Jmin, Jmax)
    H_vega_wav, H_vega_fl = cut_vega(vega_wavelength, vega_flux, Hmin, Hmax)
    K_vega_wav, K_vega_fl = cut_vega(vega_wavelength, vega_flux, Kmin, Kmax)
    W_vega_wav, W_vega_fl = cut_vega(vega_wavelength, vega_flux, Wmin, Wmax)
    
    J_vega_fl = np.array(J_vega_fl * intJ (J_vega_wav))
    H_vega_fl = np.array(H_vega_fl * intH (H_vega_wav))
    K_vega_fl = np.array(K_vega_fl * intK (K_vega_wav))
    W_vega_fl = np.array(W_vega_fl * intW (W_vega_wav))


    
    IRAC1_vega_wav, IRAC1_vega_fl = cut_vega(vega_wavelength, vega_flux, IRAC1min, IRAC1max)
    IRAC2_vega_wav, IRAC2_vega_fl = cut_vega(vega_wavelength, vega_flux, IRAC2min, IRAC2max)
    IRAC3_vega_wav, IRAC3_vega_fl = cut_vega(vega_wavelength, vega_flux, IRAC3min, IRAC3max)
    IRAC4_vega_wav, IRAC4_vega_fl = cut_vega(vega_wavelength, vega_flux, IRAC4min, IRAC4max)
    
    IRAC1_vega_fl = np.array(IRAC1_vega_fl * intIRAC1 (IRAC1_vega_wav))
    IRAC2_vega_fl = np.array(IRAC2_vega_fl * intIRAC2 (IRAC2_vega_wav))
    IRAC3_vega_fl = np.array(IRAC3_vega_fl * intIRAC3 (IRAC3_vega_wav))
    IRAC4_vega_fl = np.array(IRAC4_vega_fl * intIRAC4 (IRAC4_vega_wav))
    
    NB2_vega_wav, NB2_vega_fl = cut_vega(vega_wavelength, vega_flux, NB2min, NB2max)
    
    NB2_vega_fl = np.array(NB2_vega_fl * intNB2 (NB2_vega_wav))
    
    NB1_vega_wav, NB1_vega_fl = cut_vega(vega_wavelength, vega_flux, NB1min, NB1max)
    
    NB1_vega_fl = np.array(NB1_vega_fl * intNB1 (NB1_vega_wav))
    
    z_vega_wav, z_vega_fl = cut_vega(vega_wavelength, vega_flux, zmin, zmax)
    
    z_vega_fl = np.array(z_vega_fl * intz (z_vega_wav))
    
    J_vega_fluxes = np.trapz (J_vega_fl, J_vega_wav)
    H_vega_fluxes = np.trapz (H_vega_fl, H_vega_wav)
    K_vega_fluxes = np.trapz (K_vega_fl, K_vega_wav)
    W_vega_fluxes = np.trapz (W_vega_fl, W_vega_wav)
    
    IRAC1_vega_fluxes = np.trapz (IRAC1_vega_fl, IRAC1_vega_wav)
    IRAC2_vega_fluxes = np.trapz (IRAC2_vega_fl, IRAC2_vega_wav)
    IRAC3_vega_fluxes = np.trapz (IRAC3_vega_fl, IRAC3_vega_wav)
    IRAC4_vega_fluxes = np.trapz (IRAC4_vega_fl, IRAC4_vega_wav)
    
    NB2_vega_fluxes = np.trapz (NB2_vega_fl, NB2_vega_wav)
    NB1_vega_fluxes = np.trapz (NB1_vega_fl, NB1_vega_wav)
    z_vega_fluxes = np.trapz (z_vega_fl, z_vega_wav)
    
    J_fluxes = fluxes1(J_flux, J_wavelength)
    H_fluxes = fluxes1(H_flux, H_wavelength)
    K_fluxes = fluxes1(K_flux, K_wavelength)
    W_fluxes = fluxes1(W_flux, W_wavelength)
    
    IRAC1_fluxes = fluxes1(IRAC1_flux, IRAC1_wavelength)
    IRAC2_fluxes = fluxes1(IRAC2_flux, IRAC2_wavelength)
    IRAC3_fluxes = fluxes1(IRAC3_flux, IRAC3_wavelength)
    IRAC4_fluxes = fluxes1(IRAC4_flux, IRAC4_wavelength)
    
    NB2_fluxes = fluxes1(NB2_flux, NB2_wavelength)
    NB1_fluxes = fluxes1(NB1_flux, NB1_wavelength)
    z_fluxes = fluxes1(z_flux, z_wavelength)
    
    J_fl_ratio = J_fluxes/J_vega_fluxes
    H_fl_ratio = H_fluxes/H_vega_fluxes
    K_fl_ratio = K_fluxes/K_vega_fluxes
    W_fl_ratio = W_fluxes/W_vega_fluxes
    
    IRAC1_fl_ratio = IRAC1_fluxes/IRAC1_vega_fluxes
    IRAC2_fl_ratio = IRAC2_fluxes/IRAC2_vega_fluxes
    IRAC3_fl_ratio = IRAC3_fluxes/IRAC3_vega_fluxes
    IRAC4_fl_ratio = IRAC4_fluxes/IRAC4_vega_fluxes
    
    NB2_fl_ratio = NB2_fluxes/NB2_vega_fluxes
    NB1_fl_ratio = NB1_fluxes/NB1_vega_fluxes
    z_fl_ratio = z_fluxes/z_vega_fluxes
    
    bands = {'J' : J_fl_ratio, 'H' : H_fl_ratio, 'K' : K_fl_ratio, 'W' : W_fl_ratio, '3.6' : IRAC1_fl_ratio,\
             '4.5' : IRAC2_fl_ratio, '5.8' : IRAC3_fl_ratio, '8.0' : IRAC4_fl_ratio, \
            'NB2090' : NB2_fl_ratio, 'NB1190' : NB1_fl_ratio, 'z': z_fl_ratio}
    
        
    if band in bands:
        new_band = bands[band]
    
    mag = np.array(-2.5*np.log10(new_band))
    table_data = []
    j = 0
    

    for i in overlap4:
        row = [Model_numbers[i], Teff_vals[i], logg_vals[i], FeH_vals[i], CO_vals[i], SpT_vals[i], mag[j]]
        table_data.append(row)

        j+=1
    table_data = np.array (table_data)
    
    Table = tabulate(table_data, headers = ['Model_number', 'Teff (K)', 'logg', 'FeH', 'C/O', 'SpType', 
                                            band+' (mag)'], tablefmt='orgtbl')
    if table == True:    
        print (Table)
    return table_data



def colours (colour, Teff=50, SpT=50, FeH=50, logg=50, CO=50, table = True):
    
    split = colour.split('-')
    
    mag1 = mags (split[0], Teff, SpT, FeH, logg, CO, table= False)
    mag11 = np.array([float(i) for i in (mag1[:,6])])
    mag2 = mags (split[1], Teff, SpT, FeH, logg, CO, table= False)
    mag22 = np.array([float(i) for i in (mag2[:,6])])
    
    colour_index = np.array(mag11 - mag22)
    table_data = []
    j = 0
    for i in overlap:
        row = [Model_numbers[i], Teff_vals[i], logg_vals[i], FeH_vals[i], CO_vals[i], SpT_vals[i], colour_index[j]]
        table_data.append(row)
        j+=1
    table_data = np.array (table_data)
    
    Table = tabulate(table_data, headers = ['Model_number', 'Teff (K)', 'logg', 'FeH', 'C/O', 'SpType', colour+' (mag)'],
                     tablefmt='orgtbl')
    if table == True:    
        print (Table)
    
    return table_data


Teff = ['1000', '1250', '1500', '1750', '2000', '2250', '2500']
CO = [0.35, 0.55, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.85, 0.90, 0.91, 
      0.92, 0.93, 0.94, 0.95, 1.00, 1.05, 1.12, 1.40]