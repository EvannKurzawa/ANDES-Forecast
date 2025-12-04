#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 09 13:52:45 2024

@author: brunocanto
"""

# Copyright (C) 2024, Bruno L. Canto Martins

# ANDES ETC v2.0
# Last modification - September, 2024.
# 

# Imports
import numpy
import numpy as np
import os
from os import system, chdir
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import *
from .andes_etc_lib import *
from decimal import Decimal

from itertools import chain
from matplotlib.ticker import ScalarFormatter

##############################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))

# Constants definitions
# Physical Constants
PLANK_CONSTANT = 6.62607015e-34  # Plank's constant [J.s]
LIGHT_SPEED = 299792458.0  # Light speed [m/s]

# Telescope parameters
DTEL = 38.5  # Telescope diameter [m]
COBS = 0.28  # Fractional diameter of central obscuration of the telescope (9% in area)
FCAM = 1.5  # Focal aperture of the camera
TBCK = 283  # Ambient temperature [K]
EBCK = 0.20  # Total emissivity of the telescope and instrument
SAMPLING = 1  # # Sampling [pix]

# Instruments Requirements
RPOW = 100000 	# Resolving power
PIXBINNED = 1  #on-chip binning factor (only for CCDs - UBV and RIZ bands [lambda <= 950nm])
NDIT = 1  # Number of separte read-outs

##############################################################################
# Spectral Library
spectral_data = {
    'O5V': 1,
    'O9V': 2,
    'B0V': 3,
    'B1V': 4,
    'B3V': 5,
    'B8V': 6,
    'A0V': 7,
    'A2V': 8,
    'A3V': 9,
    'A5V': 10,
    'F0V': 11,
    'F2V': 12,
    'F5V': 13,
    'F8V': 14,
    'G0V': 15,
    'G2V': 16,
    'G5V': 17,
    'G8V': 18,
    'K0V': 19,
    'K2V': 20,
    'K5V': 21,
    'K7V': 22,
    'M0V': 23,
    'M2V': 24,
    'M4V': 25,
    'M5V': 26
}


##############################################################################
# Functions definitions
def choose_magnitude_band():
    """
    Prompt the user to choose the magnitude band.

    Returns:
    - magnitude_band (str): Chosen magnitude band.
    """
    valid_magnitude_bands = ['U', 'B', 'V', 'R', 'I', 'Z', 'Y', 'J', 'H', 'K']
    while True:
        magnitude_band = input("Choose the magnitude band (U, B, V, R, I, Z, Y, J, H, K): ").upper()
        if magnitude_band in valid_magnitude_bands:
            return magnitude_band
        else:
            print("Invalid magnitude band. Please choose between U, B, V, R, I, Z, Y, J, H or K.")

def choose_mag_system():
    """
    Prompt the user to choose between Vega and AB magnitude systems.

    Returns:
    - mag_system (str): Chosen magnitude system ('Vega' or 'AB').
    """
    while True:
        mag_system = input("Choose the magnitude system (Vega or AB): ").strip().upper()
        if mag_system == 'VEGA' or mag_system == 'AB':
            return mag_system
        else:
            print("Invalid input. Please choose between 'Vega' or 'AB'.")

def input_magnitude(magnitude_band):
    """
    Prompt the user to input the object magnitude for a specific band.

    Args:
    - magnitude_band (str): Band of the magnitude (U, B, V, R, I, Z, Y, J, H, K).

    Returns:
    - magnitude (float): Magnitude inputted by the user.
    """
    magnitude_value = float(input(f"Object magnitude in Vega system ({magnitude_band} band): "))
    return magnitude_value

def convert_vega_to_ab(magnitude_band, magnitude_value):
    """
    Convert magnitude from Vega to AB system, where the correction values are 
    based on Willmer, C.N.A. 2018, ApJS, 236, 47 
    [mips.as.arizona.edu/~cnaw/sun.html]
    
    Args:
    - magnitude_band (str): Band of the magnitude (U, B, V, R, I, Z, Y, J, H, K).
    - magnitude_vega (float): Magnitude in the Vega system.

    Returns:
    - magnitude_ab (float): Magnitude converted to the AB system.
    """

    # Dictionary with correction values for each band
    correction_values = {
        'U': 0.72,
        'B': -0.13,
        'V': -0.01,
        'R': 0.17,
        'I': 0.41,
        'Z': 0.49,
        'Y': 0.58,
        'J': 0.87,
        'H': 1.34,
        'K': 1.82
    }

    # Get the correction value for the specified band
    correction = correction_values[magnitude_band]

    # Convert magnitude from Vega to AB
    magnitude_ab = magnitude_value - correction

    return magnitude_ab

def input_specific_magnitude():
    """
    Prompt the user to choose a magnitude band and input magnitude in either Vega or AB system.

    Returns:
    - magnitude_band (str): Chosen magnitude band.
    - magnitude_value (float): Magnitude inputted by the user.
    - mag_system (str): Chosen magnitude system ('Vega' or 'AB').
    """
    magnitude_band = choose_magnitude_band()  # Prompt the user to choose a magnitude band
    mag_system = choose_mag_system()  # Prompt the user to choose the magnitude system

    # If magnitude system is AB, apply correction to the magnitude value
    if mag_system.upper() == 'AB':
        input_magnitude_value = input_magnitude(magnitude_band)
        magnitude_value = convert_vega_to_ab(magnitude_band, input_magnitude_value)
    # If magnitude system is VEGA, no conversion needed, directly input the magnitude
    else:
        input_magnitude_value = float(input(f"Object magnitude in Vega system ({magnitude_band} band): "))
        magnitude_value = input_magnitude_value

    return magnitude_band, input_magnitude_value, magnitude_value, mag_system

def input_spectral_type():
    """
    Prompt the user to choose a spectral type.

    Returns:
    - st (str): Chosen spectral type.
    """
    while True:
        st = input("Choose the spectral type (O5V/O9V/B0V/B1V/B3V/B8V/A0V/A2V/A3V/A5V/F0V/F2V/F5V/F8V/G0V/G2V/G5V/G8V/K0V/K2V/K5V/K7V/M0V/M2V/M4V/M5V): ").rstrip().upper()
        if st in spectral_data:
            return st
        else:
            print("Invalid spectral type. Please choose from the provided list.")

def calc_flux_spectral_type(st, specific_magnitude_band, specific_magnitude):
    """
    Calculate flux for a given spectral type and magnitude band, and correct magnitude to AB system.

    Args:
    - st (str): Spectral type.
    - specific_magnitude_band (str): Magnitude band (U, B, V, R, I, Z, Y, J, H, or K).
    - specific_magnitude (float): Magnitude inputted by the user.

    Returns:
    - flux_obj_UBV (list): Flux values for UBV band.
    - flux_obj_RIZ (list): Flux values for RIZ band.
    - flux_obj_YJH (list): Flux values for YJH band.
    - flux_obj_K (list): Flux values for K band.
    - z0_band_to_flux_ratio (float): Ratio of Z_0 to star flux in the specified band at lambda_nm.
    """
    # Define the folder containing spectral data
    folder_path = os.path.join(current_dir, 'Spectral_Library')    

    # Define the file names for different magnitude bands
    file_names = {
        'UBV': 'stellar_spectra_pickles_interpol_UBV_May2024.dat',
        'RIZ': 'stellar_spectra_pickles_interpol_RIZ_May2024.dat',
        'YJH': 'stellar_spectra_pickles_interpol_YJH_May2024.dat',
        'K': 'stellar_spectra_pickles_interpol_K_May2024.dat'
    }

    # Map specific subbands to their parent bands
    subband_to_parent_band = {
        'U': 'UBV',
        'B': 'UBV',
        'V': 'UBV',
        'R': 'RIZ',
        'I': 'RIZ',
        'Z': 'YJH',
        'Y': 'YJH',
        'J': 'YJH',
        'H': 'YJH',
        'K': 'K'
    }

    # Determine the correct band to use
    magnitude_band = subband_to_parent_band.get(specific_magnitude_band, specific_magnitude_band)

    # Check if the provided magnitude band is valid
    if magnitude_band not in file_names:
        raise ValueError(f"Invalid magnitude band '{specific_magnitude_band}'. Valid bands are: {list(file_names.keys()) + list(subband_to_parent_band.keys())}")

    # Extract values from ZP.dat and get lambda_ref and ZP for the chosen magnitude band
    zp_file_path = os.path.join(current_dir, "PowerLaw", "ZP.dat")
    with open(zp_file_path, 'r') as f:
        dados = {
            parts[0]: (float(parts[1]), float(parts[2]), float(parts[3]))
            for line in f
            if not line.startswith('#') and (parts := line.split())
        }
    
    lambda_ref, lambda_nm, ZP = dados.get(specific_magnitude_band, (None, None, None))
    
    if lambda_nm is None:
        raise ValueError(f"No data found for the magnitude band '{specific_magnitude_band}' in ZP.dat.")

    # Retrieve the specified spectral type
    if st in spectral_data:    
        infos = spectral_data[st] 
        index = infos

    # Initialize variable to store the flux of the star at the specific wavelength
    flux_band_value = None

    # Get the file name for the specified magnitude band
    file_name = file_names[subband_to_parent_band.get(specific_magnitude_band, specific_magnitude_band)]

    # Read the data from the appropriate file
    file_path = os.path.join(folder_path, file_name)
    lines = reading_table(file_path)

    # Extract wavelengths and fluxes from the data
    wavelengths = [row[0] for row in lines]  # Assuming first column is wavelength
    flux_sts = [row[index] for row in lines]  # Flux values for the current band
  
    # Check if the lambda_nm is present and find its index
    if lambda_nm in wavelengths:
        exact_index = wavelengths.index(lambda_nm)
        flux_band_value = flux_sts[exact_index]

    # Calculate the Z_0 to flux ratio
    if flux_band_value is not None:
        z0_band_to_flux_ratio = ZP / flux_band_value
    else:
        z0_band_to_flux_ratio = None
    
    # Initialize lists to store normalized flux values
    flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K = [], [], [], []
    
    # Apply normalization factor for each band
    for band, file_name in file_names.items():
        file_path = os.path.join(folder_path, file_name)
        lines = reading_table(file_path)
    
        # Extract wavelengths and fluxes from the data
        wavelengths = [row[0] for row in lines]  # Assuming first column is wavelength
        flux_sts = [row[index] for row in lines]  # Flux values for the current band

        # Calculate the normalized flux values
        flux = [
            (10.**(-0.4 * (specific_magnitude))) * (flux_value if flux_value != 0 else 1e-15) * 1e4 * z0_band_to_flux_ratio
            for flux_value in flux_sts
        ]
    
        if band == 'UBV':
            flux_obj_UBV.extend(flux)
        elif band == 'RIZ':
            flux_obj_RIZ.extend(flux)
        elif band == 'YJH':
            flux_obj_YJH.extend(flux)
        elif band == 'K':
            flux_obj_K.extend(flux)
        
    return flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K

def input_airmass():
    """
    Prompt the user to input the airmass value.

    The airmass represents the path length of light through the Earth's atmosphere 
    and affects the amount of atmospheric extinction experienced by the observed object.

    Returns:
    - airmass (float): The inputted airmass value.
    """
    while True:
        try:
            airmass = float(input("Airmass (range 1.00-5.55): "))
            if 1.00 <= airmass <= 5.55:
                return airmass
            else:
                print("Airmass value must be within the range 1.00-5.55.")
        except ValueError:
            print("Invalid input. Please enter a valid numeric value.")

def input_seeing():
    """
    Prompt the user to input the observation seeing.

    Seeing refers to the atmospheric turbulence affecting the sharpness 
    and stability of astronomical observations.

    Returns:
    - seeing (float): The inputted observation seeing value.
    """
    while True:
        try:
            seeing = float(input("Observation seeing (range: 0.7-1.2): "))
            return seeing
        except ValueError:
            print("Invalid input. Please enter a valid numeric value.")

def tapas_efficiency(airmass):
    """
    Calculate atmospheric transmission efficiency for different spectral bands based on TAPAS atmospheric models.

    Args:
    - airmass (float): The airmass value for the observation.

    Returns:
    - atm_eff_UBV (list): Atmospheric transmission efficiency for the UBV spectral band.
    - atm_eff_RIZ (list): Atmospheric transmission efficiency for the RIZ spectral band.
    - atm_eff_YJH (list): Atmospheric transmission efficiency for the YJH spectral band.
    - atm_eff_K (list): Atmospheric transmission efficiency for the K spectral band.
    """
    # Initialize lists to store atmospheric transmission efficiencies for each band
    atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K = [], [], [], []

    # Define the folder path containing TAPAS files
    folder_path = os.path.join(current_dir, 'TAPAS')    

    # Define file names for different spectral bands
    file_names = ['tapas_interpol_UBV_May2024.dat',
                  'tapas_interpol_RIZ_May2024.dat',
                  'tapas_interpol_YJH_May2024.dat',
                  'tapas_interpol_K_May2024.dat']

    for file_name, atm_eff_band in zip(file_names, [atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K]):
        file_path = os.path.join(folder_path, file_name)
        lines = reading_table(file_path)
    
        # Calculate atmospheric transmission efficiency based on airmass
        for flux_row in lines:
            if airmass == 1.0:
                atm_eff_band.append(flux_row[1])
            elif 1.0 < airmass < 1.02:
                atm_eff_band.append(flux_row[1] + (flux_row[2] - flux_row[1]) * (airmass - 1.00) / 0.02)
            elif airmass == 1.02:
                atm_eff_band.append(flux_row[2])
            elif 1.02 < airmass < 1.06:
                atm_eff_band.append(flux_row[2] + (flux_row[3] - flux_row[2]) * (airmass - 1.02) / 0.04)
            elif airmass == 1.06:
                atm_eff_band.append(flux_row[3])
            elif 1.06 < airmass < 1.15:
                atm_eff_band.append(flux_row[3] + (flux_row[4] - flux_row[3]) * (airmass - 1.06) / 0.09)
            elif airmass == 1.15:
                atm_eff_band.append(flux_row[4])
            elif 1.15 < airmass < 1.30:
                atm_eff_band.append(flux_row[4] + (flux_row[5] - flux_row[4]) * (airmass - 1.15) / 0.15)
            elif airmass == 1.30:
                atm_eff_band.append(flux_row[5])
            elif 1.30 < airmass < 1.55:
                atm_eff_band.append(flux_row[5] + (flux_row[6] - flux_row[5]) * (airmass - 1.30) / 0.25)
            elif airmass == 1.55:
                atm_eff_band.append(flux_row[6])
            elif 1.55 < airmass < 1.99:
                atm_eff_band.append(flux_row[6] + (flux_row[7] - flux_row[6]) * (airmass - 1.55) / 0.44)
            elif airmass == 1.99:
                atm_eff_band.append(flux_row[7])
            elif 1.99 < airmass < 2.90:
                atm_eff_band.append(flux_row[7] + (flux_row[8] - flux_row[7]) * (airmass - 1.99) / 0.91)
            elif airmass == 2.90:
                atm_eff_band.append(flux_row[8])
            elif 2.90 < airmass < 5.55:
                atm_eff_band.append(flux_row[8] + (flux_row[9] - flux_row[8]) * (airmass - 2.90) / 2.65)
            elif airmass == 5.55:
                atm_eff_band.append(flux_row[9])

    return atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K

def calc_total_efficiencies(atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K):
    """Calculate total efficiencies for different spectral types.
    
    Args:
        - atm_eff_UBV (list): Atmospheric transmission efficiencies for UBV band.
        - atm_eff_RIZ (list): Atmospheric transmission efficiencies for RIZ band.
        - atm_eff_YJH (list): Atmospheric transmission efficiencies for YJH band.
        - atm_eff_K (list): Atmospheric transmission efficiencies for K band.
        
    Returns:
        - effs_UBV (list): Total efficiencies for UBV band.
        - effs_RIZ (list): Total efficiencies for RIZ band.
        - effs_YJH (list): Total efficiencies for YJH band.
        - effs_K (list): Total efficiencies for K band.
        - effs_UBV_tel (list): Telescope efficiencies for UBV band. 
        - effs_RIZ_tel (list): Telescope efficiencies for RIZ band.
        - effs_YJH_tel (list): Telescope efficiencies for YJH band.
        - effs_K_tel (list): Telescope efficiencies for K band.
        - effs_UBV_inst (list): Instrument efficiencies (FL, FE, BE, and Detector) for UBV band.
        - effs_RIZ_inst (list): Instrument efficiencies (FL, FE, BE, and Detector) for RIZ band.
        - effs_YJH_inst (list): Instrument efficiencies (FL, FE, BE, and Detector) for YJH band.
        - effs_K_inst (list): Instrument efficiencies (FL, FE, BE, and Detector) for K band.
        - effs_UBV2 (list): Total efficiencies (without the blaze function) for UBV band.
        - effs_RIZ2 (list): Total efficiencies (without the blaze function) for RIZ band.
        - effs_YJH2 (list): Total efficiencies (without the blaze function) for YJH band.
        - effs_K2 (list): Total efficiencies (without the blaze function) for K band.
        - wavelength_UBV (list): Wavelength values for the UBV spectral band [micrometer].
        - wavelength_RIZ (list): Wavelength values for the RIZ spectral band [micrometer].
        - wavelength_YJH (list): Wavelength values for the YJH spectral band [micrometer].
        - wavelength_K (list): Wavelength values for the K spectral band [micrometer].
        
    """
    # Initialize lists to store total efficiencies and wavelengths for each band
    effs_UBV, effs_RIZ, effs_YJH, effs_K = [], [], [], []  #Total Efficiencies
    effs_UBV2, effs_RIZ2, effs_YJH2, effs_K2 = [], [], [], []  #Total Efficiencies
    effs_UBV_tel, effs_RIZ_tel, effs_YJH_tel, effs_K_tel = [], [], [], []  #Telescope Efficiencies
    effs_UBV_inst, effs_RIZ_inst, effs_YJH_inst, effs_K_inst = [], [], [], []  #Instrument Efficiencies
    effs_UBV_tel2, effs_RIZ_tel2, effs_YJH_tel2, effs_K_tel2 = [], [], [], []  #Telescope Efficiencies
    effs_UBV_inst2, effs_RIZ_inst2, effs_YJH_inst2, effs_K_inst2 = [], [], [], []  #Instrument Efficiencies
    wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K = [], [], [], []
    
    # Get the path to the 'Efficiencies' folder
    folder_path = os.path.join(current_dir, 'Efficiencies')    
    
    # List of file names for each band
    file_names = ['efficiencies_interpol_UBV_May2024.dat',
                  'efficiencies_interpol_RIZ_May2024.dat',
                  'efficiencies_interpol_YJH_May2024.dat',
                  'efficiencies_interpol_K_May2024.dat']

    # Loop through each file
    for file_name in file_names:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file_name)
        # Read the table from the file
        lines = reading_table(file_path)

        # Initialize lists to store wavelengths and total efficiencies
        wavelengths, effs_tot, effs_tel, effs_inst, effs_no_blaze = [], [], [], [], []
        
        # Loop through each row in the table
        for row in lines:
            # Extract wavelength
            wavelengths.append(row[0])
            # Calculate total efficiency by multiplying individual efficiencies
            # Telescope: row[1], FL: row[2], FE: row[3], BE: row[4], Detector: row[5], Blaze: row[6]
            eff_total = row[1] * row[2] * row[3] * row[4] * row[5] * row[6]
            eff_total2 = row[1] * row[2] * row[3] * row[4] * row[5]
            eff_total3 = row[2] * row[3] * row[4] * row[5]
            effs_tot.append(eff_total)
            effs_tel.append(row[1])
            effs_inst.append(eff_total3)
            effs_no_blaze.append(eff_total2)
            
        # Determine which band the file corresponds to and store the results accordingly
        if 'UBV' in file_name:
            wavelength_UBV = [round(wavelength/1000, 7) for wavelength in wavelengths]  #Convert wavelength in nm to micrometer
            effs_UBV.extend([eff * atm_eff for eff, atm_eff in zip(effs_tot, atm_eff_UBV)])
            effs_UBV2.extend([eff * atm_eff for eff, atm_eff in zip(effs_no_blaze, atm_eff_UBV)])
            effs_UBV_tel.extend([eff for eff in zip(effs_tel)])
            effs_UBV_inst.extend([eff for eff in zip(effs_inst)])
            effs_UBV_tel2.extend(effs_tel)
            effs_UBV_inst2.extend(effs_inst)
        elif 'RIZ' in file_name:
            wavelength_RIZ = [round(wavelength/1000, 7) for wavelength in wavelengths]  #Convert wavelength in nm to micrometer
            effs_RIZ.extend([eff * atm_eff for eff, atm_eff in zip(effs_tot, atm_eff_RIZ)])
            effs_RIZ2.extend([eff * atm_eff for eff, atm_eff in zip(effs_no_blaze, atm_eff_RIZ)])
            effs_RIZ_tel.extend([eff for eff in zip(effs_tel)])
            effs_RIZ_inst.extend([eff for eff in zip(effs_inst)])
            effs_RIZ_tel2.extend(effs_tel)
            effs_RIZ_inst2.extend(effs_inst)
        elif 'YJH' in file_name:
            wavelength_YJH = [round(wavelength/1000, 7) for wavelength in wavelengths]  #Convert wavelength in nm to micrometer
            effs_YJH.extend([eff * atm_eff for eff, atm_eff in zip(effs_tot, atm_eff_YJH)])
            effs_YJH2.extend([eff * atm_eff for eff, atm_eff in zip(effs_no_blaze, atm_eff_YJH)])
            effs_YJH_tel.extend([eff for eff in zip(effs_tel)])
            effs_YJH_inst.extend([eff for eff in zip(effs_inst)])
            effs_YJH_tel2.extend(effs_tel)
            effs_YJH_inst2.extend(effs_inst)
        elif 'K' in file_name:
            wavelength_K = [round(wavelength/1000, 7) for wavelength in wavelengths]  #Convert wavelength in nm to micrometer
            effs_K.extend([eff * atm_eff for eff, atm_eff in zip(effs_tot, atm_eff_K)])
            effs_K2.extend([eff * atm_eff for eff, atm_eff in zip(effs_no_blaze, atm_eff_K)])
            effs_K_tel.extend([eff for eff in zip(effs_tel)])
            effs_K_inst.extend([eff for eff in zip(effs_inst)])
            effs_K_tel2.extend(effs_tel)
            effs_K_inst2.extend(effs_inst)
    
    # Return total efficiencies and corresponding wavelengths for all bands
    return effs_UBV, effs_RIZ, effs_YJH, effs_K, effs_UBV_tel, effs_RIZ_tel, effs_YJH_tel, effs_K_tel, effs_UBV_inst, effs_RIZ_inst, effs_YJH_inst, effs_K_inst, effs_UBV2, effs_RIZ2, effs_YJH2, effs_K2, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K

def calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, parameter):
    """
    Calculates the aperture values (DAPE [circular aperture]), the pixel size (DPIX [m]),
    the read-out noise (RON [e-/pix]), and the dark current (DARKCUR [e-/pix/hr]) based 
    on the wavelength and parameter provided. 
    """
    if parameter == 'dape':
        # Calculate aperture values based on the wavelength.
        DAPE_UBV = [0.87 if w <= 0.95 else 0.71 for w in wavelength_UBV]
        DAPE_RIZ = [0.87 if w <= 0.95 else 0.71 for w in wavelength_RIZ]
        DAPE_YJH = [0.87 if w <= 0.95 else 0.71 for w in wavelength_YJH]
        DAPE_K = [0.87 if w <= 0.95 else 0.71 for w in wavelength_K]

        return DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K

    elif parameter == 'dpix':
        # Calculate pixel size values based on the wavelength.
        DPIX_UBV = [10 if w <= 0.95 else 15 for w in wavelength_UBV]
        DPIX_RIZ = [10 if w <= 0.95 else 15 for w in wavelength_RIZ]
        DPIX_YJH = [10 if w <= 0.95 else 15 for w in wavelength_YJH]
        DPIX_K = [10 if w <= 0.95 else 15 for w in wavelength_K]

        return DPIX_UBV, DPIX_RIZ, DPIX_YJH, DPIX_K

    elif parameter == 'ron':
        # Calculate read-out noise (RON) based on the wavelength.
        RON_UBV = [1 if w <= 0.95 else 4.5 for w in wavelength_UBV]
        RON_RIZ = [1 if w <= 0.95 else 4.5 for w in wavelength_RIZ]
        RON_YJH = [1 if w <= 0.95 else 4.5 for w in wavelength_YJH]
        RON_K = [1 if w <= 0.95 else 4.5 for w in wavelength_K]

        return RON_UBV, RON_RIZ, RON_YJH, RON_K

    elif parameter == 'darkcur':
        # Calculate dark current (DARKCUR) based on the wavelength.
        DARKCUR_UBV = [1 if w <= 0.95 else 20 for w in wavelength_UBV]
        DARKCUR_RIZ = [1 if w <= 0.95 else 20 for w in wavelength_RIZ]
        DARKCUR_YJH = [1 if w <= 0.95 else 20 for w in wavelength_YJH]
        DARKCUR_K = [1 if w <= 0.95 else 20 for w in wavelength_K]

        return DARKCUR_UBV, DARKCUR_RIZ, DARKCUR_YJH, DARKCUR_K

    elif parameter == 'bin_size':
        # Calculate bin size based on the wavelength [Angstrom].
        # resolution=WL/RPOW [m] 
        bin_size_UBV = [((w*1e-6)/RPOW)*1e10/SAMPLING for w in wavelength_UBV]
        bin_size_RIZ = [((w*1e-6)/RPOW)*1e10/SAMPLING for w in wavelength_RIZ]
        bin_size_YJH = [((w*1e-6)/RPOW)*1e10/SAMPLING for w in wavelength_YJH]
        bin_size_K = [((w*1e-6)/RPOW)*1e10/SAMPLING for w in wavelength_K]

        return bin_size_UBV, bin_size_RIZ, bin_size_YJH, bin_size_K
        
    elif parameter == 'photon':
        # Calculate energy of one photon based on the wavelength [ergs].
        photon_UBV = [(PLANK_CONSTANT*LIGHT_SPEED/(w*1e-6))*1e7 for w in wavelength_UBV]
        photon_RIZ = [(PLANK_CONSTANT*LIGHT_SPEED/(w*1e-6))*1e7 for w in wavelength_RIZ]
        photon_YJH = [(PLANK_CONSTANT*LIGHT_SPEED/(w*1e-6))*1e7 for w in wavelength_YJH]
        photon_K = [(PLANK_CONSTANT*LIGHT_SPEED/(w*1e-6))*1e7 for w in wavelength_K]
        
        return photon_UBV, photon_RIZ, photon_YJH, photon_K

    else:
        raise ValueError("Invalid parameter. Please specify either 'dape', 'dpix', 'ron', or 'darkcur'.")
    
def calc_telescope_area(DTEL, COBS):
    """
    Calculate the area of the telescope aperture.

    Args:
    - DTEL (float): Telescope diameter (m).
    - COBS (float): Fractional diameter of central obscuration of the telescope.

    Returns:
    - ATEL (float): Area of the telescope aperture (m^2).
    """
    ATEL = numpy.pi / 4 * 10000 * DTEL ** 2 * (1 - COBS ** 2)  # Calculate the telescope area [cm^2]
    
    return ATEL

def input_exposure_time():
    t_exp = float(input("Exposure time (in sec): "))  # Exposure time in seconds
    return t_exp

def process_band_parameters(DTEL, FCAM, PIXBINNED, NDIT, EXPTIME, DPIX, DAPE, RON, DARKCUR):
    """
    Process parameters for a specific spectral band.

    Args:
    - DTEL (float): Telescope diameter.
    - FCAM (float): Focal aperture of the camera.
    - PIXBINNED (float): Binning factor of pixels (only for lambda < 0.95 micrometers).
    - NDIT (int): Number of separate read-outs.
    - EXPTIME (float): Exposure time in seconds.
    - DPIX (list): List of pixel sizes for the band.
    - DAPE (list): List of sky-projected angular diameters for the band.
    - RON (list): List of read-out noise values for the band.
    - DARKCUR (list): List of dark current values for the band.

    Returns:
    - ANPIX (list): Pixel sky projected angular size [arcsec] for each band.
    - PIXAPE (list): Number of pixels corresponding to spectrometer aperture [pixels] for each band.
    - TOTPIXRE (list): Total number of read-out pixels per resolution element for each band.
    - NOISEDET (list): Noise in the detector for each band.
    """
    # Initialize empty lists to store results for the current band
    ANPIX, PIXAPE, TOTPIXRE, NOISEDET = [], [], [], []
    
    # Iterate over the zipped arguments for the current band
    for i in range(len(DPIX)):
        # Extract individual parameters for the current band
        DPIX_val, DAPE_val, RON_val, DARKCUR_val = DPIX[i], DAPE[i], RON[i], DARKCUR[i]
        
        # Calculate parameters for the current band
        ANPIX_val = DPIX_val * 1e-6 / DTEL / FCAM * 3600 * 180 / np.pi  # Pixel sky projected angular size [arcsec]
        PIXAPE_val = (np.pi / 4 * DAPE_val ** 2) / (ANPIX_val ** 2)  # Number of pixels corresponding to spectrometer aperture [pixels]
        TOTPIXRE_val = int(PIXAPE_val / PIXBINNED)  # Total number of read-out pixels per resolution element
        NOISEDET_val = np.sqrt(PIXAPE_val * (NDIT * RON_val ** 2 / PIXBINNED + DARKCUR_val * EXPTIME/3600))  # electrons [e-]
        
        # Append calculated parameters to the corresponding lists for the current band
        ANPIX.append(ANPIX_val)
        PIXAPE.append(PIXAPE_val)
        TOTPIXRE.append(TOTPIXRE_val)
        NOISEDET.append(NOISEDET_val)
        
    # Return lists for each parameter for the current band
    return ANPIX, PIXAPE, TOTPIXRE, NOISEDET

def sky_background(airmass):
    """
    Calculate sky background contribution for different spectral bands based on Table 3 from ANDES ETC Fortran version.

    Args:
    - airmass (float): The airmass value for the observation.

    Returns:
    - sky_background_UBV (list): Sky background contribution for the UBV spectral band.
    - sky_background_RIZ (list): Sky background contribution for the RIZ spectral band.
    - sky_background_YJH (list): Sky background contribution for the YJH spectral band.
    - sky_background_K (list): Sky background contribution for the K spectral band.
    """
    # Initialize lists to store sky background contribution for each band
    sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K = [], [], [], []
    

    # Define the folder path containing Sky Background files
    folder_path = os.path.join(current_dir, 'Sky_Background')    

    # Define file names for different spectral bands
    file_names = ['sky_interpol_UBV_May2024.dat',
                  'sky_interpol_RIZ_May2024.dat',
                  'sky_interpol_YJH_May2024.dat',
                  'sky_interpol_K_May2024.dat']

    for file_name, sky_background_band in zip(file_names, [sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K]):
        file_path = os.path.join(folder_path, file_name)
        lines = reading_table(file_path)
    
        # Calculate atmospheric transmission efficiency based on airmass [AB mag/arcsec^2]
        for flux_row in lines:
            if airmass == 1.0:
                sky_background_band.append(flux_row[1])
            elif 1.0 < airmass < 1.02:
                sky_background_band.append(flux_row[1] + (flux_row[2] - flux_row[1]) * (airmass - 1.00) / 0.02)
            elif airmass == 1.02:
                sky_background_band.append(flux_row[2])
            elif 1.02 < airmass < 1.06:
                sky_background_band.append(flux_row[2] + (flux_row[3] - flux_row[2]) * (airmass - 1.02) / 0.04)
            elif airmass == 1.06:
                sky_background_band.append(flux_row[3])
            elif 1.06 < airmass < 1.15:
                sky_background_band.append(flux_row[3] + (flux_row[4] - flux_row[3]) * (airmass - 1.06) / 0.09)
            elif airmass == 1.15:
                sky_background_band.append(flux_row[4])
            elif 1.15 < airmass < 1.30:
                sky_background_band.append(flux_row[4] + (flux_row[5] - flux_row[4]) * (airmass - 1.15) / 0.16)
            elif airmass == 1.30:
                sky_background_band.append(flux_row[5])
            elif 1.30 < airmass < 1.55:
                sky_background_band.append(flux_row[5] + (flux_row[6] - flux_row[5]) * (airmass - 1.30) / 0.25)
            elif airmass == 1.55:
                sky_background_band.append(flux_row[6])
            elif 1.55 < airmass < 1.99:
                sky_background_band.append(flux_row[6] + (flux_row[7] - flux_row[6]) * (airmass - 1.55) / 0.44)
            elif airmass == 1.99:
                sky_background_band.append(flux_row[7])
            elif 1.99 < airmass < 2.90:
                sky_background_band.append(flux_row[7] + (flux_row[8] - flux_row[7]) * (airmass - 1.99) / 0.91)
            elif airmass == 2.90:
                sky_background_band.append(flux_row[8])
            elif 2.90 < airmass < 5.55:
                sky_background_band.append(flux_row[8] + (flux_row[9] - flux_row[8]) * (airmass - 2.90) / 2.65)
            elif airmass == 5.55:
                sky_background_band.append(flux_row[9])

    return sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K

def calculate_background_flux(RPOW, EBCK, TBCK, ATEL, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, effs_UBV, effs_RIZ, effs_YJH, effs_K, DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K, sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K, EXPTIME):
    """
    Calculate the background flux and corresponding noise for different spectral bands.

    Args:
    - RPOW (float): Resolution power.
    - EBCK (float): Background emission level.
    - TBCK (float): Background temperature.
    - ATEL (float): Telescope area.
    - wavelength_UBV (list): Wavelength values for the UBV spectral band.
    - wavelength_RIZ (list): Wavelength values for the RIZ spectral band.
    - wavelength_YJH (list): Wavelength values for the YJH spectral band.
    - wavelength_K (list): Wavelength values for the K spectral band.
    - effs_UBV (list): List of total efficiencies for the UBV spectral band.
    - effs_RIZ (list): List of total efficiencies for the RIZ spectral band.
    - effs_YJH (list): List of total efficiencies for the YJH spectral band.
    - effs_K (list): List of total efficiencies for the K spectral band.
    - DAPE_UBV (list): List of aperture diameters for the UBV spectral band.
    - DAPE_RIZ (list): List of aperture diameters for the RIZ spectral band.
    - DAPE_YJH (list): List of aperture diameters for the YJH spectral band.
    - DAPE_K (list): List of aperture diameters for the K spectral band.
    - sky_background_UBV (list): Sky background values for the UBV spectral band.
    - sky_background_RIZ (list): Sky background values for the RIZ spectral band.
    - sky_background_YJH (list): Sky background values for the YJH spectral band.
    - sky_background_K (list): Sky background values for the K spectral band.

    Returns:
    - NBCK_UBV (list): Background flux per second for the UBV spectral band.
    - NOISEBCK_UBV (list): Noise in the background flux per second for the UBV spectral band.
    - NBCK_RIZ (list): Background flux per second for the RIZ spectral band.
    - NOISEBCK_RIZ (list): Noise in the background flux per second for the RIZ spectral band.
    - NBCK_YJH (list): Background flux per second for the YJH spectral band.
    - NOISEBCK_YJH (list): Noise in the background flux per second for the YJH spectral band.
    - NBCK_K (list): Background flux per second for the K spectral band.
    - NOISEBCK_K (list): Noise in the background flux per second for the K spectral band.
    """
    NBCK_UBV = []
    NBCK_RIZ = []
    NBCK_YJH = []
    NBCK_K = []
    NOISEBCK_UBV = []
    NOISEBCK_RIZ = []
    NOISEBCK_YJH = []
    NOISEBCK_K = []

    for i in range(len(wavelength_UBV)):
        sigmask = (10**((16.85 - sky_background_UBV[i]) / 2.5)) / RPOW  # [ph cm-2 s-1 arcsec-2]
        sigmath = (1.4e12 * EBCK * np.exp(-14388. / (wavelength_UBV[i] * TBCK))) / (wavelength_UBV[i]**3 * RPOW)  # [ph cm-2 s-1 arcsec-2]
        NBCK_UBV.append(effs_UBV[i] * ATEL * np.pi / 4 * DAPE_UBV[i]**2 * (sigmask + sigmath))  # [e-/s]
        NOISEBCK_UBV.append(np.sqrt(NBCK_UBV[-1] * EXPTIME))

    for i in range(len(wavelength_RIZ)):
        sigmask = (10**((16.85 - sky_background_RIZ[i]) / 2.5)) / RPOW  # [ph cm-2 s-1 arcsec-2]
        sigmath = (1.4e12 * EBCK * np.exp(-14388. / (wavelength_RIZ[i] * TBCK))) / (wavelength_RIZ[i]**3 * RPOW)  # [ph cm-2 s-1 arcsec-2]
        NBCK_RIZ.append(effs_RIZ[i] * ATEL * np.pi / 4 * DAPE_RIZ[i]**2 * (sigmask + sigmath))  # [e-/s]
        NOISEBCK_RIZ.append(np.sqrt(NBCK_RIZ[-1] * EXPTIME))

    for i in range(len(wavelength_YJH)):
        sigmask = (10**((16.85 - sky_background_YJH[i]) / 2.5)) / RPOW  # [ph cm-2 s-1 arcsec-2]
        sigmath = (1.4e12 * EBCK * np.exp(-14388. / (wavelength_YJH[i] * TBCK))) / (wavelength_YJH[i]**3 * RPOW)  # [ph cm-2 s-1 arcsec-2]
        NBCK_YJH.append(effs_YJH[i] * ATEL * np.pi / 4 * DAPE_YJH[i]**2 * (sigmask + sigmath))  # [e-/s]
        NOISEBCK_YJH.append(np.sqrt(NBCK_YJH[-1] * EXPTIME))

    for i in range(len(wavelength_K)):
        sigmask = (10**((16.85 - sky_background_K[i]) / 2.5)) / RPOW  # [ph cm-2 s-1 arcsec-2]
        sigmath = (1.4e12 * EBCK * np.exp(-14388. / (wavelength_K[i] * TBCK))) / (wavelength_K[i]**3 * RPOW)  # [ph cm-2 s-1 arcsec-2]
        NBCK_K.append(effs_K[i] * ATEL * np.pi / 4 * DAPE_K[i]**2 * (sigmask + sigmath))  # [e-/s]
        NOISEBCK_K.append(np.sqrt(NBCK_K[-1] * EXPTIME))

    return NBCK_UBV, NOISEBCK_UBV, NBCK_RIZ, NOISEBCK_RIZ, NBCK_YJH, NOISEBCK_YJH, NBCK_K, NOISEBCK_K

def calculate_mAB(flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K, photon_UBV, photon_RIZ, photon_YJH, photon_K, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K):
    """
    Calculate object signal per resolution element on the detector.

    Args:
    - flux_obj_UBV (list): List of corrected flux for UBV band.
    - flux_obj_RIZ (list): List of corrected flux for RIZ band.
    - flux_obj_YJH (list): List of corrected flux for YJH band.
    - flux_obj_K (list): List of corrected flux for K band.
    - photon_UBV (list): List of photon energy for UBV band [ergs].
    - photon_RIZ (list): List of photon energy for RIZ band [ergs].
    - photon_YJH (list): List of photon energy for YJH band [ergs].
    - photon_K (list): List of photon energy for K band [ergs].

    Returns:
    - mAB_UBV (list): Object signal per resolution element for the UBV spectral band [e-].
    - mAB_RIZ (list): Object signal per resolution element for the RIZ spectral band [e-].
    - mAB_YJH (list): Object signal per resolution element for the YJH spectral band [e-].
    - mAB_K (list): Object signal per resolution element for the K spectral band [e-].
    """
    # Initialize lists to store object signal for each band
    mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K = [], [], [], []

    # Define the folder path containing TAPAS files
    folder_path = os.path.join(current_dir, 'Slit_Efficiency')    

    # Define file names for different spectral bands
    file_names = ['slit_efficiency_UBV_May2024.dat',
                  'slit_efficiency_RIZ_May2024.dat',
                  'slit_efficiency_YJH_May2024.dat',
                  'slit_efficiency_K_May2024.dat']

    # Loop through each file
    for file_name in file_names:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file_name)
        # Read the table from the file
        lines = reading_table(file_path)
    
        # Determine which band the file corresponds to and store the results accordingly
        if 'UBV' in file_name:
            mAB_band = []
            for i in range(len(wavelength_UBV)):
                mAB_band.append(-2.5 * np.log10(flux_obj_UBV[i] / photon_UBV[i]) - 2.5 * np.log10(wavelength_UBV[i]) + 16.85)
            mAB_UBV.extend(mAB_band)
        elif 'RIZ' in file_name:
            mAB_band = []
            for i in range(len(wavelength_RIZ)):
                mAB_band.append(-2.5 * np.log10(flux_obj_RIZ[i] / photon_RIZ[i]) - 2.5 * np.log10(wavelength_RIZ[i]) + 16.85)
            mAB_RIZ.extend(mAB_band)
        elif 'YJH' in file_name:
            mAB_band = []
            for i in range(len(wavelength_YJH)):
                mAB_band.append(-2.5 * np.log10(flux_obj_YJH[i] / photon_YJH[i]) - 2.5 * np.log10(wavelength_YJH[i]) + 16.85)
            mAB_YJH.extend(mAB_band)
        elif 'K' in file_name:
            mAB_band = []
            for i in range(len(wavelength_K)):
                mAB_band.append(-2.5 * np.log10(flux_obj_K[i] / photon_K[i]) - 2.5 * np.log10(wavelength_K[i]) + 16.85)
            mAB_K.extend(mAB_band)

    return mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K

def obj_signal(ATEL, EXPTIME, RPOW, effs_UBV, effs_RIZ, effs_YJH, effs_K, mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K):
    """
    Calculate object signal per resolution element on the detector.

    Args:
    - ATEL (float): Telescope area.
    - EXPTIME (float): Exposure time.
    - RPOW (float): Resolution power.
    - effs_UBV (list): List of total efficiencies for UBV spectral band.
    - effs_RIZ (list): List of total efficiencies for RIZ spectral band.
    - effs_YJH (list): List of total efficiencies for YJH spectral band.
    - effs_K (list): List of total efficiencies for K spectral band.
    - mAB_UBV (list): List of photon energy for UBV band [ergs].
    - mAB_RIZ (list): List of photon energy for RIZ band [ergs].
    - mAB_YJH (list): List of photon energy for YJH band [ergs].
    - mAB_K (list): List of photon energy for K band [ergs].

    Returns:
    - nobj_UBV (list): Object signal per resolution element for the UBV spectral band [e-].
    - nobj_RIZ (list): Object signal per resolution element for the RIZ spectral band [e-].
    - nobj_YJH (list): Object signal per resolution element for the YJH spectral band [e-].
    - nobj_K (list): Object signal per resolution element for the K spectral band [e-].
    """
    
    # Initialize lists to store object signal for each band
    nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K = [], [], [], []

    # Define the folder path containing TAPAS files
    folder_path = os.path.join(current_dir, 'Slit_Efficiency')    

    # Define file names for different spectral bands
    file_names = ['slit_efficiency_UBV_May2024.dat',
                  'slit_efficiency_RIZ_May2024.dat',
                  'slit_efficiency_YJH_May2024.dat',
                  'slit_efficiency_K_May2024.dat']

    # Initialize variables to store slit efficiencies for each band
    slit_UBV, slit_RIZ, slit_YJH, slit_K = [], [], [], []

    # Loop through each file
    for file_name in file_names:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file_name)
        # Read the table from the file
        lines = reading_table(file_path)
    
        # Initialize list to store slit efficiencies
        slit_band = []
        # Loop through each row in the table
        for row in lines:
            # Extract wavelength
            slit_band.append(row[0]/100)
        
        # Determine which band the file corresponds to and store the results accordingly
        if 'UBV' in file_name:
            slit_UBV = slit_band[:]
            nobj_band = []
            for i in range(len(effs_UBV)):
                nobj_band.append((slit_UBV[i] * effs_UBV[i] * ATEL * EXPTIME / RPOW) * (10**((16.85 - mAB_UBV[i]) / 2.5)))
            nobj_UBV.extend(nobj_band)
        elif 'RIZ' in file_name:
            slit_RIZ = slit_band[:]
            nobj_band = []
            for i in range(len(effs_RIZ)):
                nobj_band.append((slit_RIZ[i] * effs_RIZ[i] * ATEL * EXPTIME / RPOW) * (10**((16.85 - mAB_RIZ[i]) / 2.5)))
            nobj_RIZ.extend(nobj_band)
        elif 'YJH' in file_name:
            slit_YJH = slit_band[:]
            nobj_band = []
            for i in range(len(effs_YJH)):
                nobj_band.append((slit_YJH[i] * effs_YJH[i] * ATEL * EXPTIME / RPOW) * (10**((16.85 - mAB_YJH[i]) / 2.5)))
            nobj_YJH.extend(nobj_band)
        elif 'K' in file_name:
            slit_K = slit_band[:]
            nobj_band = []
            for i in range(len(effs_K)):
                nobj_band.append((slit_K[i] * effs_K[i] * ATEL * EXPTIME / RPOW) * (10**((16.85 - mAB_K[i]) / 2.5)))
            nobj_K.extend(nobj_band)

    return nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K

def SN_obj(nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K, NOISEBCK_UBV, NOISEBCK_RIZ, NOISEBCK_YJH, NOISEBCK_K, NOISEDET_UBV, NOISEDET_RIZ, NOISEDET_YJH, NOISEDET_K, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K):
    """
    Calculate signal-to-noise per resolution element on the detector.

    Args:
    - nobj_UBV (list): Object signal per resolution element for the UBV spectral band.
    - nobj_RIZ (list): Object signal per resolution element for the RIZ spectral band.
    - nobj_YJH (list): Object signal per resolution element for the YJH spectral band.
    - nobj_K (list): Object signal per resolution element for the K spectral band.
    - NOISEBCK_UBV (list): Noise in the background flux per second for the UBV spectral band.
    - NOISEBCK_RIZ (list): Noise in the background flux per second for the RIZ spectral band.
    - NOISEBCK_YJH (list): Noise in the background flux per second for the YJH spectral band.
    - NOISEBCK_K (list): Noise in the background flux per second for the K spectral band.
    - NOISEDET_UBV (list): Noise in the detector for the UBV spectral band.
    - NOISEDET_RIZ (list): Noise in the detector for the RIZ spectral band.
    - NOISEDET_YJH (list): Noise in the detector for the YJH spectral band.
    - NOISEDET_K (list): Noise in the detector for the K spectral band.

    Returns:
    - SN_UBV (list): Signal-to-noise per resolution element for the UBV spectral band.
    - SN_RIZ (list): Signal-to-noise per resolution element for the RIZ spectral band.
    - SN_YJH (list): Signal-to-noise per resolution element for the YJH spectral band.
    - SN_K (list): Signal-to-noise per resolution element for the K spectral band.
    """
    # Initialize lists to store object signal for each band
    SN_UBV = [np.sqrt((nobj**2)/(NOISEBCK**2+NOISEDET**2+nobj)) for nobj, NOISEBCK, NOISEDET in zip(nobj_UBV, NOISEBCK_UBV, NOISEDET_UBV)]
    SN_RIZ = [np.sqrt((nobj**2)/(NOISEBCK**2+NOISEDET**2+nobj)) for nobj, NOISEBCK, NOISEDET in zip(nobj_RIZ, NOISEBCK_RIZ, NOISEDET_RIZ)]
    SN_YJH = [np.sqrt((nobj**2)/(NOISEBCK**2+NOISEDET**2+nobj)) for nobj, NOISEBCK, NOISEDET in zip(nobj_YJH, NOISEBCK_YJH, NOISEDET_YJH)]
    SN_K = [np.sqrt((nobj**2)/(NOISEBCK**2+NOISEDET**2+nobj)) for nobj, NOISEBCK, NOISEDET in zip(nobj_K, NOISEBCK_K, NOISEDET_K)]

    return SN_UBV, SN_RIZ, SN_YJH, SN_K



def summon_ETC(spectral_type: str, exptime: float, mag_value: float):
    # Fixed inputs
    mag_band = "J"
    # system = "Vega"
    airmass = 1.0

    # Inform the user of the chosen defaults
    # print(f"Running ETC with mag_band={mag_band}, system={system}, airmass={airmass}")
    print(f"Running ETC with mag_band={mag_band}, airmass={airmass}")
    print(f"Spectral type: {spectral_type}, Exposure time: {exptime} s")

    ATEL = calc_telescope_area(DTEL, COBS)

    flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K = calc_flux_spectral_type(spectral_type, mag_band, mag_value)

    atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K = tapas_efficiency(airmass)

    effs_UBV, effs_RIZ, effs_YJH, effs_K, effs_UBV_tel, effs_RIZ_tel, effs_YJH_tel, effs_K_tel, effs_UBV_inst, effs_RIZ_inst, effs_YJH_inst, effs_K_inst, effs_UBV2, effs_RIZ2, effs_YJH2, effs_K2, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K = calc_total_efficiencies(atm_eff_UBV, atm_eff_RIZ, atm_eff_YJH, atm_eff_K)

    DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'dape')
    DPIX_UBV, DPIX_RIZ, DPIX_YJH, DPIX_K = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'dpix')
    RON_UBV, RON_RIZ, RON_YJH, RON_K = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'ron')
    DARKCUR_UBV, DARKCUR_RIZ, DARKCUR_YJH, DARKCUR_K = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'darkcur')
    bin_size_UBV, bin_size_RIZ, bin_size_YJH, bin_size_K  = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'bin_size')
    photon_UBV, photon_RIZ, photon_YJH, photon_K  = calculate_instrument_parameter_values(wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, 'photon')

    ANPIX_UBV, PIXAPE_UBV, TOTPIXRE_UBV, NOISEDET_UBV = process_band_parameters(DTEL, FCAM, PIXBINNED, NDIT, exptime, DPIX_UBV, DAPE_UBV, RON_UBV, DARKCUR_UBV)
    ANPIX_RIZ, PIXAPE_RIZ, TOTPIXRE_RIZ, NOISEDET_RIZ = process_band_parameters(DTEL, FCAM, PIXBINNED, NDIT, exptime, DPIX_RIZ, DAPE_RIZ, RON_RIZ, DARKCUR_RIZ)
    ANPIX_YJH, PIXAPE_YJH, TOTPIXRE_YJH, NOISEDET_YJH = process_band_parameters(DTEL, FCAM, PIXBINNED, NDIT, exptime, DPIX_YJH, DAPE_YJH, RON_YJH, DARKCUR_YJH)
    ANPIX_K, PIXAPE_K, TOTPIXRE_K, NOISEDET_K = process_band_parameters(DTEL, FCAM, PIXBINNED, NDIT, exptime, DPIX_K, DAPE_K, RON_K, DARKCUR_K)

    sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K = sky_background(airmass)

    NBCK_UBV, NOISEBCK_UBV, NBCK_RIZ, NOISEBCK_RIZ, NBCK_YJH, NOISEBCK_YJH, NBCK_K, NOISEBCK_K = calculate_background_flux(RPOW, EBCK, TBCK, ATEL, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K, effs_UBV, effs_RIZ, effs_YJH, effs_K, DAPE_UBV, DAPE_RIZ, DAPE_YJH, DAPE_K, sky_background_UBV, sky_background_RIZ, sky_background_YJH, sky_background_K, exptime)
    
    mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K = calculate_mAB(flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K, photon_UBV, photon_RIZ, photon_YJH, photon_K, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K)

    nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K = obj_signal(ATEL, exptime, RPOW, effs_UBV, effs_RIZ, effs_YJH, effs_K, mAB_UBV, mAB_RIZ, mAB_YJH, mAB_K)

    SN_UBV, SN_RIZ, SN_YJH, SN_K = SN_obj(nobj_UBV, nobj_RIZ, nobj_YJH, nobj_K, NOISEBCK_UBV, NOISEBCK_RIZ, NOISEBCK_YJH, NOISEBCK_K, NOISEDET_UBV, NOISEDET_RIZ, NOISEDET_YJH, NOISEDET_K, wavelength_UBV, wavelength_RIZ, wavelength_YJH, wavelength_K)

    wavelengths = [np.array(wavelength_UBV) * 1000, np.array(wavelength_RIZ) * 1000, np.array(wavelength_YJH) * 1000, np.array(wavelength_K) * 1000]
    SN = [SN_UBV, SN_RIZ, SN_YJH, SN_K]
    stellar_fluxes = [flux_obj_UBV, flux_obj_RIZ, flux_obj_YJH, flux_obj_K]

    return wavelengths, SN, stellar_fluxes
