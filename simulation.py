import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.config import petitradtrans_config_parser
from PyAstronomy.pyasl import dopplerShift
from tqdm import tqdm
import sys
import os
from scipy import ndimage
from PyAstronomy.pyasl import instrBroadGaussFast
from petitRADTRANS.chemistry.utils import volume_mixing_ratios2mass_fractions
import petitRADTRANS.physical_constants as cst
from ETC.Custom_ETC import summon_ETC
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ProcessPoolExecutor, as_completed
import secrets





""" =========================================== STEP 1 ===========================================

Retrieving the parameters of the simulation from a big .csv table which contains one line per planet, and all the data.
This table is called in the bash scripts and used as system argument in this code. """


if len(sys.argv) != 14:
    sys.exit(1)

planet_name = sys.argv[1]
semi_major_axis = float(sys.argv[2]) * cst.au            #cm
inclination = float(sys.argv[3]) * np.pi / 180           #rad
orbital_period = float(sys.argv[4]) * 24 * 3600          #s
transit_duration = float(sys.argv[5]) * 3600             #s
T_eq = float(sys.argv[6])
radius = float(sys.argv[7]) * cst.r_earth                #cm
star_radius = float(sys.argv[8]) * cst.r_sun             #cm
reference_gravity = float(sys.argv[9]) * 100
reference_pressure = 1.013
transit_time = float(sys.argv[10])
star_spectral_type = sys.argv[11]
J_magnitude = float(sys.argv[12])
v_tot = float(sys.argv[13])                              #km





""" =========================================== STEP 2 ===========================================

Next comes all the functions definitions.

1) main: Calls the new python ETC, and stacks the outputs together to return wavelengths, S/N ratios and stellar fluxes.

2) build_kernel_from_template: Builds a .npy file of a molecule's autocorrelation, used as a kernel in further Bayesian analysis

3) add_noise_to_flux: Given a noiseless flux and a S/N ratio array obtained for a specific exposure time, it returns the associated noisy flux and variance.

4) continuum_normalization: Performs a division-continuum-normalization on a given flux.

5) cross_correlation_analysis: Performs a cross-correlation between one noisy and one noiseless flux.
A normalization by σ² is included.
The function doesn't return anything but exports the CCF as a .npy file, for further Bayesian analysis. 

6) vacuum_to_air: Converts vacuum wavelengths (from petitRADTRANS) into air wavelengths (adapted for the ETC)."""


def main():
    exptimes = [5 * transit_time, 10 * transit_time, 20 * transit_time, 40 * transit_time]
    sn = {}

    # run the new ETC
    wavelengths_list, SN_list, stellar_fluxes = summon_ETC(spectral_type=star_spectral_type, exptime=600, mag_value=J_magnitude)  #exptime = 10min here

    # concatenate the four bands UBV → RIZ → YJH → K
    all_wl = np.concatenate(wavelengths_list) * 1e-7                              #convert in cms
    sort_idx = np.argsort(all_wl)                                                 # As the edges of the four band wv overlap, we sort it correctly
    all_wl_sorted = all_wl[sort_idx]

    for idx, exptime in enumerate(exptimes, start=1):
        all_sn = np.concatenate(SN_list) * np.sqrt(exptime/600)                   # I called the ETC with 10min so i multiply by sqrt(my_reql_exptime/600).

        all_sn_sorted = all_sn[sort_idx]                                          # SN accordingly matches the WV order

        sn[f"exp_{idx}"] = all_sn_sorted                                          # store for different exptimes

    whole_stellar_flux = np.concatenate(stellar_fluxes)
    whole_stellar_flux_sorted = whole_stellar_flux[sort_idx]                      # Stellar flux accordingly matches the WV order

    return all_wl_sorted, sn, whole_stellar_flux_sorted



def build_kernel_from_template(resampled_wavelengths, noiseless_template, CCF_velocities, mol_name):
    # 1) Recreate the same edge mask I use in cross_correlation_analysis
    shifted_left,  _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_template, CCF_velocities[0])
    shifted_right, _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_template, CCF_velocities[-1])
    mask_left  = ~np.isnan(shifted_left)
    mask_right = ~np.isnan(shifted_right)
    mask_ccf_edges = np.logical_and(mask_left, mask_right)

    # 2) Reference vector = unshifted template, masked, mean-subtracted (mirrors the "noisy_middle_tr" prep)
    unshifted = noiseless_template[mask_ccf_edges]
    unshifted -= np.mean(unshifted)

    # 3) Looping over Δv, shifting template the same way I do for the CCF template,
    #    normalize to unit sum then mean-subtract (mirrors your template normalization)
    K = []
    for dv in CCF_velocities:
        shifted, _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_template, dv)
        shifted = shifted[mask_ccf_edges]

        shifted /= np.sum(shifted)
        shifted -= np.mean(shifted)

        # No variance weighting here: we want the shape only
        K.append(np.sum(shifted * unshifted))

    K = np.asarray(K)

    # 4) Normalize to unit peak for a clean kernel core
    peak = np.max(np.abs(K))
    K /= peak

    # 5) Save
    kdir = os.path.join(".", "kernels")
    os.makedirs(kdir, exist_ok=True)
    np.save(os.path.join(kdir, f"{mol_name}_kernel.npy"), {'v': CCF_velocities, 'k': K})

    return K



def add_noise_to_flux(flux, sn_ratio):
    flux = np.array(flux, copy=False)
    sigma = flux / sn_ratio
    noisy_flux = np.random.normal(loc=flux, scale=sigma)
    return noisy_flux, sigma**2


def continuum_normalization(arr):                                                 # Aaron's super nice function!
    sigma = 510
    truncate = 2
    gauss = arr.copy()
    gauss[np.isnan(gauss)] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0, truncate=truncate)
 
    norm = np.ones(shape=arr.shape)
    norm[np.isnan(arr)] = 0
    norm = ndimage.gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0, truncate=truncate)

    norm = np.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[np.isnan(arr)] = np.nan

    normalized_flux = arr / gauss                                                  # We chose a continuum-normalization by division and not subtraction.
    
    return normalized_flux, gauss


def cross_correlation_analysis(noisy_flux, noiseless_flux, flux_variance, ccf_iter_number, res_directory, exp_label):
    CC1d = []

    # 1) We forecast the worst NaNs on the left, on the right, and remove them in every iteration. It remains piddling (around 100 NaNs in total for 300 000 values)
    shifted_left, _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_flux, CCF_velocities[0])
    shifted_right, _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_flux, CCF_velocities[-1])

    mask_left = ~np.isnan(shifted_left)
    mask_right = ~np.isnan(shifted_right)
    mask_ccf_edges = np.logical_and(mask_left, mask_right)


    # 2) We apply the mask to the noisy flux, to the variance array, and later in the CCF loop, to the noiseless template. "tr" stands for "truncated".
    noisy_middle_tr = noisy_flux[mask_ccf_edges]
    variance_tr = flux_variance[mask_ccf_edges]

    # 3) Variance array normalization
    variance_tr /= np.sum(variance_tr)
    noisy_middle_tr -= np.mean(noisy_middle_tr)

    # 5) Main CCF loop: doppler shift, mask applied on the noiseless shifted template, normalization of the template, and finally, the cross-correlation
    for i, velocity in enumerate(CCF_velocities):
        CC_exp = []
        shifted_noiseless_flux, _ = dopplerShift(resampled_wavelengths * 1e8, noiseless_flux, velocity)

        shifted_noiseless_flux_tr = shifted_noiseless_flux[mask_ccf_edges]

        shifted_noiseless_flux_tr /= np.sum(shifted_noiseless_flux_tr)                                             # Normalization of the template
        shifted_noiseless_flux_tr -= np.mean(shifted_noiseless_flux_tr)

        CC_exp   = np.sum(shifted_noiseless_flux_tr * noisy_middle_tr / variance_tr)                               # The CC is here
        CC1d.append(CC_exp)

    CC1d = np.asarray(CC1d)

    exp_dir = os.path.join(res_directory, str(exp_label))
    os.makedirs(exp_dir, exist_ok=True)
    cc_filename = os.path.join(exp_dir, f"CC1d_{mol_studied}_{ccf_iter_number}.npy")
    np.save(cc_filename, CC1d)




def vacuum_to_air(wavelength_vacuum_cm):
    wl_vac_AA = wavelength_vacuum_cm * 1e8  # convert cm to Å

    # Edlén formula valid between ~2000–25000 Å
    sigma2 = (1e4 / wl_vac_AA) ** 2  # (μm^-2)
    n = 1 + 0.0000834254 + 0.02406147 / (130 - sigma2) + 0.00015998 / (38.9 - sigma2)

    wl_air_AA = wl_vac_AA / n
    wl_air_cm = wl_air_AA * 1e-8  # back to cm

    return wl_air_cm




""" =========================================== STEP 3 ===========================================

Setting up useful constants and parameters.

1) "computationnal_quality" defines the number of orbital phases we will use.
A big number would be wiser to produce fancy 2D plots, but reducing it to 4 (or 5, i tried and it is the same) doesn't change the results.
I chose 4 so as to have 1 phase before transit and 1 after so 2 phases out-of-transit, and 2 during transits.

2) "sn_percentage_removed" is the threshold below which we mask the pixels in the original ETC S/N array.
We choose to remove the smallest/most chaotic 3% of the pixels.

3) "CCF_velocities" are used in the very end to build the CCF (see the CCF function)

4) If an orbital phase is between minus "phase_in" and "phase_in", then it is in transit.

5) One wavelength array is built, four S/N arrays for the four different exposure times, and one stellar flux.
They are built by calling the main() function which calls itself the ETC.
The "keep_mask" is then applied. This mask is applied to each array just mentionned.

6) The stellar flux alone is Doppler-shifted by v_tot (systemic velocity - barycentric correction). The shift gives NaNs on the edges, which are, again, masked for each array.

7) Convenient directories are created, along with a file containing v_tot, for further Bayesian analysis."""


computationnal_quality = 50
sn_percentage_removed = 3
CCF_velocities = np.linspace(-200, 200, 401)                                                         #km/s
petitradtrans_config_parser.set_input_data_path('/export/home/kurzawa/petitRADTRANS/input_data')
Kp_known = (2 * np.pi * semi_major_axis * np.sin(inclination)) / orbital_period                       #cm/s
Kp_known /= 100000                                                                                    #km/s
times = 2 * transit_duration * (np.linspace(0, 1, computationnal_quality) - 0.5)
orbital_phases = times / orbital_period
phase_in = transit_duration / orbital_period / 2
mol_array = ["H2O", "O2", "CO2", "CH4"]


WL_ETC, sn_dict, star_spectrum = main()
SN_1_ETC = sn_dict['exp_1']
SN_2_ETC = sn_dict['exp_2']
SN_3_ETC = sn_dict['exp_3']
SN_4_ETC = sn_dict['exp_4']


# We take the 3rd percentile, so 3% of the smallest values, in the ordered arrays
thr1 = np.percentile(SN_1_ETC, sn_percentage_removed)
thr2 = np.percentile(SN_2_ETC, sn_percentage_removed)
thr3 = np.percentile(SN_3_ETC, sn_percentage_removed)
thr4 = np.percentile(SN_4_ETC, sn_percentage_removed)


# We build boolean masks of "good" pixels
m1 = SN_1_ETC > thr1
m2 = SN_2_ETC > thr2
m3 = SN_3_ETC > thr3
m4 = SN_4_ETC > thr4

keep_mask = m1 & m2 & m3 & m4

SN_1_ETC_no_zero          = SN_1_ETC[keep_mask]
SN_2_ETC_no_zero          = SN_2_ETC[keep_mask]
SN_3_ETC_no_zero          = SN_3_ETC[keep_mask]
SN_4_ETC_no_zero          = SN_4_ETC[keep_mask]
WL_ETC_no_zero            = WL_ETC[keep_mask]
F_star_resampled_uncut    = star_spectrum[keep_mask]


F_star_resampled_shifted, _ = dopplerShift(WL_ETC_no_zero * 1e8, F_star_resampled_uncut, v_tot)
star_mask = ~np.isnan(F_star_resampled_shifted)
F_star_shifted_cut = F_star_resampled_shifted[star_mask]
wv_star_masked = WL_ETC_no_zero[star_mask]
SN_1_star_masked = SN_1_ETC_no_zero[star_mask]
SN_2_star_masked = SN_2_ETC_no_zero[star_mask]
SN_3_star_masked = SN_3_ETC_no_zero[star_mask]
SN_4_star_masked = SN_4_ETC_no_zero[star_mask]



safe_planet = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in planet_name)

output_dir = os.path.join("results", safe_planet)
os.makedirs(output_dir, exist_ok=True)

vtot_filename = f"v_tot_{safe_planet}.txt"
vtot_path     = os.path.join(output_dir, vtot_filename)

with open(vtot_path, "w") as vf:
    vf.write(f"{v_tot}\n")

print(f"Wrote total velocity ({v_tot}) to {vtot_path}")



""" =========================================== STEP 4 ===========================================

This part of the code creates the petitRADTRANS model, then calculates the transit depth, then broadens the spectrum, and finally resamples it to ANDES resolution."""


for mol_index, mol_studied in enumerate(mol_array):
    print(f"Starting simulation for {mol_studied}...")

    prefix = os.path.join(output_dir, f"results_{safe_planet}_{mol_studied}")
                                                                                                     #expected in microns                 #expected in bars, not baryes!!
    radtrans = Radtrans(line_species=[mol_studied], rayleigh_species = ['N2'], wavelength_boundaries=[0.3, 2.5], line_opacity_mode='lbl', pressures=np.logspace(-7, 0, 100))

    p = np.logspace(-7, 0, 100)
    p_trop  = 0.1                 # bar

    kappa = 0.17
    
    T_skin = (2.0 ** -0.25) * T_eq
    T_1bar = T_skin * (1.0 / p_trop) ** kappa
    temperatures = np.empty_like(p)
    strato = p <= p_trop
    tropos = p >  p_trop

    temperatures[strato] = T_skin
    temperatures[tropos] = T_skin * (p[tropos] / p_trop) ** kappa

    x_H2O_strat = 3.0e-6   # 3 ppmv
    x_H2O_cap   = 1.0e-2

    def saturation_vapor_pressure_bar(T):
        #Clausius–Clapeyron over liquid water. Returns saturation vapor pressure in bar.
        # e0 at T0=273.15 K
        e0_Pa, T0, Lv, Rv = 611.2, 273.15, 2.5e6, 461.0  # Pa, K, J/kg, J/kg/K
        e_Pa = e0_Pa * np.exp((Lv/Rv) * (1.0/T0 - 1.0/np.maximum(T, 150.0)))
        return e_Pa / 1.0e5  # bar

    x_H2O = np.full_like(p, x_H2O_strat)

    qs = saturation_vapor_pressure_bar(temperatures) / p
    x_H2O[tropos] = np.minimum(qs[tropos], x_H2O_cap)

    p_peak    = 0.01      # bar (10 mbar)
    sigma_ln  = 0.5       # width in ln(p)
    peak_ppm = 10.0
    O3_peak  = peak_ppm * 1e-6
    x_O3 = O3_peak * np.exp(-0.5 * (np.log(p/p_peak) / sigma_ln)**2)
    x_O3 = np.maximum(x_O3, 1.0e-8)  # ~10 ppb floor

    x_CH4_trop = 1.9e-6
    x_CH4 = np.where(p > p_trop, x_CH4_trop, x_CH4_trop * (p/p_trop)**0.5)

    x_CO2 = 4.2e-4 * np.ones_like(p)
    x_O2  = 0.2095 * np.ones_like(p)
    # CO: a bit less aloft
    x_CO  = np.where(p > p_trop, 1.0e-7, 5.0e-8)

    # N2 fills the remainder; then normalization of each layer to sum to 1
    minor_sum = x_O2 + x_O3 + x_CO2 + x_CO + x_CH4 + x_H2O
    x_N2 = 1.0 - minor_sum
    x_N2 = np.clip(x_N2, 0.0, None)

    VMR = {
        'N2':  x_N2,
        'O2':  x_O2,
        'O3':  x_O3,
        'CO2': x_CO2,
        'CO':  x_CO,
        'CH4': x_CH4,
        'H2O': x_H2O,
    }

    # Renormalize (in case of small roundoff)
    total = np.zeros_like(p)
    for s in VMR: total += VMR[s]
    for s in VMR: VMR[s] = VMR[s] / total

    # Mean molecular massin g/mol (layer-by-layer)
    M = {'N2':28.0134,'O2':31.9988,'O3':47.9982,'CO2':44.0095,'CO':28.0101,'CH4':16.043,'H2O':18.01528}
    mean_molar_masses = np.zeros_like(p, dtype=float)
    for s in VMR:
        mean_molar_masses += VMR[s] * M[s]   # μ = Σ(x_i * M_i)

    mass_fractions = volume_mixing_ratios2mass_fractions(VMR, mean_molar_masses)

    # wavelengths in cm
    wavelengths, transit_radii, _ = radtrans.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        planet_radius=radius,
        reference_pressure=reference_pressure
    )


    # The transit depth is broadened to R=100,000 and then resampled to match the wavelength array returned by the ETC (instead of using petitRADTRANS's one)
    depth = (transit_radii**2 / star_radius**2)

    broadened_depth = instrBroadGaussFast(wavelengths * 1e8, depth, 100000, edgeHandling="firstlast", maxsig=5, equid=True)

    wavelengths_air = vacuum_to_air(wavelengths)

    resampled_depth = np.interp(WL_ETC_no_zero, wavelengths_air, broadened_depth)
    resampled_depth_star_masked = resampled_depth[star_mask]



    """ =========================================== STEP 5 ===========================================

    This part of the code builds the flux by injecting the Doppler-shifted planetary signal into the star flux. Then it adds the noise corresponding to each exposure time.
    After this step, the result is four noisy fluxes. """


    # Here again we use a mask because doppler-shifting results in NaNs on the edges. As some velocities produce more NaNs that others, we apply the "worst mask" to everyone, regardless of the iteration.
    # To to this, every mask is added to this table and we then take the union. Here again, it is a small number of NaNs compared to the number of datapoints.
    masks_first_doppler = []
    F_transit_shifted_uncut = []
    F_transit_shifted = []

    for phase in orbital_phases:
        temp_shift = np.sin(2 * np.pi * phase) * Kp_known                                           #km/s

        depth_shifted, _ = dopplerShift(wv_star_masked * 1e8, resampled_depth_star_masked, temp_shift + v_tot)  # Doppler-shift by the planetary velocity + v_sys + v_bary

        mask = ~np.isnan(depth_shifted)                                                             # True where depth_shifted_Nan isn't NaN

        masks_first_doppler.append(mask)
        
        if -phase_in <= phase <= phase_in:                                                          # Injection of the planetary signal only during transit period
            temp_table = F_star_shifted_cut * (1 - depth_shifted)
            F_transit_shifted_uncut.append(temp_table)
        else:
            F_transit_shifted_uncut.append(F_star_shifted_cut)

    mask_first_doppler = np.logical_and.reduce(masks_first_doppler)


    # In these next lines, we apply the mask to the related arrays.
    for i in range(computationnal_quality):
        F_transit_shifted.append(F_transit_shifted_uncut[i][mask_first_doppler])

    resampled_wavelengths = wv_star_masked[mask_first_doppler]
    depth_masked = resampled_depth_star_masked[mask_first_doppler]

    noiseless_depth = 1 - depth_masked
    F_noiseless_normalized, _ = continuum_normalization(noiseless_depth)


    # To launch just once to create the kernels, in the first run, then it'll be used for every planets: here I create the autocorrelation kernels for each 4 molecules.
    build_kernel_from_template(resampled_wavelengths=resampled_wavelengths, noiseless_template=F_noiseless_normalized, CCF_velocities=CCF_velocities, mol_name=mol_studied)

    SN_1 = SN_1_star_masked[mask_first_doppler]
    SN_2 = SN_2_star_masked[mask_first_doppler]
    SN_3 = SN_3_star_masked[mask_first_doppler]
    SN_4 = SN_4_star_masked[mask_first_doppler]

    def init_worker(F_ts, S1, S2, S3, S4, rw, dm, phases, ccf_vel, cq, Kp, vt, pname, mstud, tdur, step):
        global F_transit_shifted, SN_1, SN_2, SN_3, SN_4, resampled_wavelengths, depth_masked
        global orbital_phases, CCF_velocities, computationnal_quality, Kp_known, v_tot
        global planet_name, mol_studied, transit_duration, transit_time, filename
        F_transit_shifted = F_ts
        SN_1 = S1
        SN_2 = S2
        SN_3= S3
        SN_4 = S4
        resampled_wavelengths = rw
        depth_masked = dm
        orbital_phases = phases
        CCF_velocities = ccf_vel
        computationnal_quality = cq
        Kp_known = Kp
        v_tot = vt
        planet_name = pname
        mol_studied = mstud
        transit_duration = tdur
        transit_time = step


    def worker(iter_idx):
        np.random.seed(secrets.randbits(32) ^ (iter_idx + 0x9E3779B9))
        # Adding the noise to the flux, and saving the related σ²
        noisy_flux_1, variance1 = add_noise_to_flux(F_transit_shifted.copy(), SN_1)
        noisy_flux_2, variance2 = add_noise_to_flux(F_transit_shifted.copy(), SN_2)
        noisy_flux_3, variance3 = add_noise_to_flux(F_transit_shifted.copy(), SN_3)
        noisy_flux_4, variance4 = add_noise_to_flux(F_transit_shifted.copy(), SN_4)


        """ =========================================== STEP 6 ===========================================

        This is the step of all the normalizations.
        1) Normalizing the obtained noisy fluxes (and their variance) by their mean. REMOVED 
        2) Normalizing the flux and its variance by the out-of-transit components, using an accurate analytical model.
        3) Continuum-normalizing the flux.
        4) Normalizing the flux by its median, along rows (to remove horizontal stripes).
        5) Continuum-normalizing a noiseless model, to prepare it for cross-correlation with the noisy flux.
        6) Normalizing σ² by the continuum of the flux
        7) Normalizing σ² by the row-wise median. """


        # Substep 1:
        mid = computationnal_quality // 2
        pre  = np.arange(computationnal_quality // 4)
        post = np.arange(computationnal_quality - computationnal_quality // 4, computationnal_quality)   # last q rows
        oot  = np.concatenate([pre, post])
        N_oot = oot.size

        # Substep 2:
        M1    = np.mean(noisy_flux_1[oot], axis=0, keepdims=True)
        M2    = np.mean(noisy_flux_2[oot], axis=0, keepdims=True)
        M3    = np.mean(noisy_flux_3[oot], axis=0, keepdims=True)
        M4    = np.mean(noisy_flux_4[oot], axis=0, keepdims=True)

        flux_pre1 = noisy_flux_1.copy()
        var_pre1  = variance1.copy()

        flux_pre2 = noisy_flux_2.copy()
        var_pre2  = variance2.copy()

        flux_pre3 = noisy_flux_3.copy()
        var_pre3  = variance3.copy()

        flux_pre4 = noisy_flux_4.copy()
        var_pre4  = variance4.copy()


        noisy_flux_1 = flux_pre1 / M1
        noisy_flux_2 = flux_pre2 / M2
        noisy_flux_3 = flux_pre3 / M3
        noisy_flux_4 = flux_pre4 / M4

        var_M1 = np.sum(var_pre1[oot], axis=0, keepdims=True) / (N_oot**2)
        var_M2 = np.sum(var_pre2[oot], axis=0, keepdims=True) / (N_oot**2)
        var_M3 = np.sum(var_pre3[oot], axis=0, keepdims=True) / (N_oot**2)
        var_M4 = np.sum(var_pre4[oot], axis=0, keepdims=True) / (N_oot**2)

        var_covariance_term1        = np.zeros_like(var_pre1)
        var_covariance_term1[oot]   = (-2.0 * flux_pre1[oot] / (M1**3 * N_oot)) * var_pre1[oot]

        var_covariance_term2        = np.zeros_like(var_pre2)
        var_covariance_term2[oot]   = (-2.0 * flux_pre2[oot] / (M2**3 * N_oot)) * var_pre2[oot]

        var_covariance_term3        = np.zeros_like(var_pre3)
        var_covariance_term3[oot]   = (-2.0 * flux_pre3[oot] / (M3**3 * N_oot)) * var_pre3[oot]

        var_covariance_term4        = np.zeros_like(var_pre4)
        var_covariance_term4[oot]   = (-2.0 * flux_pre4[oot] / (M4**3 * N_oot)) * var_pre4[oot]

        variance1 = (var_pre1 / M1**2) + (flux_pre1**2) * (var_M1 / M1**4) + var_covariance_term1
        variance2 = (var_pre2 / M2**2) + (flux_pre2**2) * (var_M2 / M2**4) + var_covariance_term2
        variance3 = (var_pre3 / M3**2) + (flux_pre3**2) * (var_M3 / M3**4) + var_covariance_term3
        variance4 = (var_pre4 / M4**2) + (flux_pre4**2) * (var_M4 / M4**4) + var_covariance_term4

        variance1_mid = variance1[mid]
        variance2_mid = variance2[mid]
        variance3_mid = variance3[mid]
        variance4_mid = variance4[mid]


        noisy_mid_transit_1 = noisy_flux_1[computationnal_quality // 2].copy()
        noisy_mid_transit_2 = noisy_flux_2[computationnal_quality // 2].copy()
        noisy_mid_transit_3 = noisy_flux_3[computationnal_quality // 2].copy()
        noisy_mid_transit_4 = noisy_flux_4[computationnal_quality // 2].copy()


        # Substep 3:
        F_continuum_normalized_1, continuum1 = continuum_normalization(noisy_mid_transit_1)
        F_continuum_normalized_2, continuum2 = continuum_normalization(noisy_mid_transit_2)
        F_continuum_normalized_3, continuum3 = continuum_normalization(noisy_mid_transit_3)
        F_continuum_normalized_4, continuum4 = continuum_normalization(noisy_mid_transit_4)


        # Substep 4:
        median1 = np.median(F_continuum_normalized_1)
        median2 = np.median(F_continuum_normalized_2)
        median3 = np.median(F_continuum_normalized_3)
        median4 = np.median(F_continuum_normalized_4)

        F_continuum_normalized_1 /= median1
        F_continuum_normalized_2 /= median2
        F_continuum_normalized_3 /= median3
        F_continuum_normalized_4 /= median4



        # Substep 6:
        variance1_mid /= continuum1**2
        variance2_mid /= continuum2**2
        variance3_mid /= continuum3**2
        variance4_mid /= continuum4**2
        
        # Substep 7:
        variance1_mid /= median1**2
        variance2_mid /= median2**2
        variance3_mid /= median3**2
        variance4_mid /= median4**2




        """ =========================================== STEP 7 ===========================================

        Final step of this code: the cross-correlation.
        Once performed, a .npy file is exported in the right folder, and will be analysed with a second script, to measure detection."""


        # Every argument used in the function corresponds to a mid-transit observation. F_noiseless_normalized is a mid-transit flux, and the variance and F_continuum_normalized_i are evaluated at the mid-phase.
        cross_correlation_analysis(F_continuum_normalized_1, F_noiseless_normalized, variance1_mid, iter_idx, output_dir, "1")
        cross_correlation_analysis(F_continuum_normalized_2, F_noiseless_normalized, variance2_mid, iter_idx, output_dir, "2")
        cross_correlation_analysis(F_continuum_normalized_3, F_noiseless_normalized, variance3_mid, iter_idx, output_dir, "3")
        cross_correlation_analysis(F_continuum_normalized_4, F_noiseless_normalized, variance4_mid, iter_idx, output_dir, "4")


        return iter_idx


     # launch pool
    initargs = (
        F_transit_shifted, SN_1, SN_2, SN_3, SN_4,
        resampled_wavelengths, depth_masked,
        orbital_phases, CCF_velocities,
        computationnal_quality, Kp_known, v_tot,
        planet_name, mol_studied,
        transit_duration, transit_time
    )
    with ProcessPoolExecutor(max_workers=10,
                             initializer=init_worker,
                             initargs=initargs) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        for future in tqdm(as_completed(futures), total=100):
            try:
                future.result()
            except Exception as e:
                print("Worker failed with:", e)

    print(f"Simulation for {mol_studied} completed.\n")