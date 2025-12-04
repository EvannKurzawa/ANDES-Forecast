#!/usr/bin/env python3
import os, glob, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless for workers
import matplotlib.pyplot as plt
import pymultinest
from pymultinest import Analyzer
from scipy.stats import norm
from scipy.special import erfinv
from collections import Counter
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, as_completed


# Config variables
dir_results = "results"
dir_output = "multinest_output_BBE"
velocity_range = (-200, 200)  # km/s
tot_velocity_pts = 401
EXP_TIMES = ["1", "2", "3", "4"]


# Off-peak penalty controls
ETA_BASE   = 0.4
GAMMA  = 1.5
REL_CAP    = 3.0
OFF_PEAK_WINDOW = 50.0


# This function returns all the subdirectories of "results" (Which planets do we have results for?)
def list_safe_planets(results_dir):
    planets = []
    for entry in os.listdir(results_dir):
        full_path = os.path.join(results_dir, entry)
        if os.path.isdir(full_path):
            planets.append(entry)
    return planets


# This function returns the molecules for which a planet has CCF results in the subdirectories analyzed in list_safe_planets
def list_molecules(results_dir, safe_planet, exp_time=None):
    base = os.path.join(results_dir, safe_planet)
    if exp_time is not None:
        base = os.path.join(base, str(exp_time))

    path = os.path.join(base, "CC1d_*_*.npy")
    files = glob.glob(path)

    molecules = []
    for filename in files:
        name = os.path.basename(filename)
        parts = name.split("_")
        if len(parts) >= 3:
            molecule = parts[1]
            if molecule not in molecules:
                molecules.append(molecule)

    molecules.sort()
    return molecules


# This function loads the kernels generated with the first script: one for each molecule.
def load_kernel_arrays(molecule):
    path = os.path.join("kernels", f"{molecule}_kernel.npy")
    if not os.path.exists(path):
        return None, None

    raw = np.load(path, allow_pickle=True)

    # Here we unwrap 0-D object arrays saved via np.save(dict, ...)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    # We accept either dict {'v','k'} or bare 1D array
    if isinstance(raw, dict):
        kv = np.asarray(raw.get('v'), dtype=float)
        ky = np.asarray(raw.get('k'), dtype=float)
    elif isinstance(raw, np.ndarray):
        ky = np.asarray(raw, dtype=float)
        kv = np.linspace(velocity_range[0], velocity_range[1], ky.size)  # km/s
    else:
        raise TypeError(f"Unsupported kernel file format: {type(raw)}")

    return kv, ky


# This function returns the full signal model (baseline + kernel core), based on prior arguments
# Core = A * K( (v - v_tot - (mu - v_tot)) / kappa ), with K given by (kernel_v, kernel_y), unit-peak.
def signal_model_kernel(velocity_array, mu, A, y0, b, c, v_tot, kernel_v, kernel_y, kappa):
    x = (velocity_array - v_tot)
    baseline = y0 + b*x + c*(x**2)

    delta = (mu - v_tot)
    arg = (velocity_array - v_tot - delta) / max(kappa, 1e-9)

    # Translation of this line: Look up the kernel curve at position (v-µ/𝜅), interpolate if necessary, force it to 0 outside the known domain, and then scale the whole kernel by the amplitude A
    core = A * np.interp(arg, kernel_v, kernel_y, left=0.0, right=0.0)
    return baseline + core


# This function translates the Bayes factor result into a Jeffrey's scale interpretation (log version)
def jeffreys_category(log10B):
    if log10B < 0:        return 'supports null'
    elif log10B < 0.5:    return 'negligible'
    elif log10B < 1.0:    return 'substantial'
    elif log10B < 1.5:    return 'strong'
    elif log10B < 2.0:    return 'very strong'
    else:                 return 'decisive'


# This function calculates the median absolute deviation of the off-peak part of the CCF
def offpeak_mad(y, v, v_tot):
    m = np.abs(v - v_tot) > OFF_PEAK_WINDOW
    if m.sum() < 8:
        return 0.0
    z = y[m] - np.median(y[m])
    return np.median(np.abs(z)), z


# This function computes the empirical lag-1 autocorrelation of a sequence (corresponds to the Pearson correlation coefficient between the sequence shifted by 1 step)
# It is winsorized by 3MAD, and clipped.
def estimate_rho_offpeak(y, v, v_tot):
    mad, z = offpeak_mad(y, v, v_tot)
    zw = np.clip(z, -3.0 * mad, 3.0 * mad)

    num = np.dot(zw[1:], zw[:-1])
    den = np.dot(zw, zw)
    r = num / den
    return float(np.clip(r, -0.9, 0.9))


# This function builds the complex terms of the log-likelihood function: the quadratic factor (qf) and the logarithm of the determinant of the correlation matrix R.
def ar1_loglike_terms(z, rho):
    n = z.size
    r2 = rho * rho
    one_minus_r2 = 1.0 - r2
    d = z[1:] - rho * z[:-1]
    qf = z[0]*z[0] + np.dot(d, d) / one_minus_r2
    logdetR = (n - 1) * np.log(one_minus_r2)
    return qf, float(logdetR)


# This function keeps all prior-cube values strictly between 0 and 1, avoiding infinities when transforming them into physical parameters with norm.ppf for instance
def clip_cube(cube, ndim):
    for i in range(ndim):
        if cube[i] <= 0.0:
            cube[i] = 1e-9
        elif cube[i] >= 1.0:
            cube[i] = 1.0 - 1e-9


# This function tells MultiNest how to turn the uniform [0,1] cube into physical parameters for the signal model
# Each parameter gets a prior, reflecting what we expect physically
# Params: [0]=mu, [1]=A, [2]=kappa, [3]=y0, [4]=b, [5]=c
def signal_priors(sigma_ccf_scalar, v_tot, A_sigma):
    dv_span = float(velocity_range[1] - velocity_range[0])     # Full velocity range
    b_sigma = (0.2 * A_sigma) / dv_span                        # Std on the baseline slope b
    c_sigma = (0.1 * A_sigma) / (dv_span**2)                   # Std on the baseline curve c

    def prior(cube, ndim, nparams):                            # "nparams" is useless but needed in the function definition to respect the standard form MultiNest expects, and avoid any error
        clip_cube(cube, ndim)

        # mu ~ N(v_tot, 5 km/s)
        mu_sigma = 5
        cube[0] = norm.ppf(cube[0], loc=v_tot, scale=mu_sigma)

        # A ~ HalfNormal(A_sigma)
        cube[1] = A_sigma * sqrt(2.0) * erfinv(cube[1])

        # kappa ~ log-uniform [0.5, 2.0]
        kmin, kmax = 0.5, 2.0
        u = cube[2]
        cube[2] = kmin * ((kmax / kmin) ** u)

        # Baseline y0, b, c
        cube[3] = norm.ppf(cube[3], loc=0.0, scale=sigma_ccf_scalar)
        cube[4] = norm.ppf(cube[4], loc=0.0, scale=b_sigma)
        cube[5] = norm.ppf(cube[5], loc=0.0, scale=c_sigma)
    return prior


# Same than signal_priors(), but for the flat model.
# The flat model shares the same baseline as the signal, but with no core.
# Params: [0]=y0, [1]=b, [2]=c
def flat_priors(sigma_ccf_scalar, A_sigma):
    dv_span = float(velocity_range[1] - velocity_range[0])
    b_sigma = (0.2 * A_sigma) / dv_span
    c_sigma = (0.1 * A_sigma) / (dv_span**2)

    def prior(cube, ndim, nparams):
        clip_cube(cube, ndim)
        cube[0] = norm.ppf(cube[0], loc=0.0, scale=sigma_ccf_scalar)
        cube[1] = norm.ppf(cube[1], loc=0.0, scale=b_sigma)
        cube[2] = norm.ppf(cube[2], loc=0.0, scale=c_sigma)
    return prior


# This function returns the likelihood function required for MultiNest (signal case)
def loglike_signal_function(velocity_array, y, std_ccf, v_tot, kernel_v, kernel_y):
    op_mad, _ = offpeak_mad(y, velocity_array, v_tot)
    rel = min(op_mad / np.median(std_ccf), REL_CAP)
    eta_eff = ETA_BASE * (1.0 + GAMMA * (rel**2))
    std_eff = std_ccf * np.sqrt(1.0 + eta_eff**2)
    rho = estimate_rho_offpeak(y, velocity_array, v_tot)
    sum_log_std = np.sum(np.log(std_eff))

    def loglike(cube, ndim, nparams):      # Again, useless arguments but needed in MultiNest standards
        mu, A, kappa, y0, b, c = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
        model = signal_model_kernel(velocity_array, mu, A, y0, b, c, v_tot, kernel_v, kernel_y, kappa)
        z = (y - model) / std_eff
        qf, logdetR = ar1_loglike_terms(z, rho)
        return -0.5 * (qf + logdetR) - sum_log_std
    return loglike


# This function returns the likelihood function required for MultiNest (flat case). Same baseline, but no core component
def loglike_flat_function(velocity_array, y, std_ccf, v_tot):
    op_mad, _ = offpeak_mad(y, velocity_array, v_tot)
    rel = min(op_mad / np.median(std_ccf), REL_CAP)
    eta_eff = ETA_BASE * (1.0 + GAMMA * (rel**2))
    std_eff = std_ccf * np.sqrt(1.0 + eta_eff**2)
    rho = estimate_rho_offpeak(y, velocity_array, v_tot)
    sum_log_std = np.sum(np.log(std_eff))

    def loglike(cube, ndim, nparams):
        y0, b, c = cube[0], cube[1], cube[2]
        x = (velocity_array - v_tot)
        model = y0 + b*x + c*(x**2)
        z = (y - model) / std_eff
        qf, logdetR = ar1_loglike_terms(z, rho)
        return -0.5 * (qf + logdetR) - sum_log_std
    return loglike


# This function opens the MultiNest results for a given model, looks up the global log-evidence and its error (trying several possible key names), and returns it as a Python float.
def read_logZ_and_err(prefix, n_params):
    an = Analyzer(n_params=n_params, outputfiles_basename=prefix)             # We initialize an Analyzer (Analyzer is a PyMultiNest helper class that reads the saved output files for a run)
    st = an.get_stats()                                                       # This loads a Python dictionary with lots of run statistics, including the global log-evidence.

    # Accepting multiple possible key names for evidence and its error
    Z_KEYS   = ('nested sampling global log-evidence', 'global evidence', 'logZ', 'log_evidence')
    ERR_KEYS = ('nested sampling global log-evidence error', 'global evidence error', 'logZerr', 'log_evidence_error')

    def pick(d, keys):                                                        # Return d[k] for the first k present in keys, else None.
        for k in keys:
            if k in d:
                return d[k]
        return None

    logZ    = pick(st, Z_KEYS)                                                # Try top level
    logZerr = pick(st, ERR_KEYS)

    if logZ is None or logZerr is None:                                       # If not found, search one level deeper (some versions nest under 'nested sampling' etc.)
        for v in st.values():
            if isinstance(v, dict):
                if logZ    is None: logZ    = pick(v, Z_KEYS)
                if logZerr is None: logZerr = pick(v, ERR_KEYS)

    if logZ is None:                                                          # Validate and convert to floats; if error missing, return NaN for it
        raise RuntimeError(f"Evidence not found in stats for {prefix}")
    return logZ, logZerr


# This function plots relevant data following the use of the framework, per ccf. Namely:
# -The CCF
# -The best-fit (MAP)
# -The prior
# -The posterior of amplitude A
# -The posterior of centroid μ
# -The evidence score
def sanity_plots(velocity_array, CCF_set, std_ccf, post_sig, prior_signal_typical, prior_flat_typical, v_tot, out_png, main_title, kernel_v, kernel_y):
    bestfit_curve = None
    if post_sig.size > 0:
        if post_sig.shape[1] >= 7 and np.isfinite(post_sig[:, -1]).any():
            idx_best = int(np.nanargmax(post_sig[:, -1]))
            p = post_sig[idx_best, :6]
        else:
            p = np.median(post_sig[:, :6], axis=0)
        bestfit_curve = signal_model_kernel(velocity_array, p[0], p[1], p[3], p[4], p[5], v_tot, kernel_v, kernel_y, kappa=p[2])

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # Top-left: CCF + best-fit
    axs[0,0].plot(velocity_array, CCF_set, label='CCF (this realization)')
    axs[0,0].fill_between(velocity_array, (CCF_set-std_ccf), (CCF_set+std_ccf),
                          alpha=0.2, label='CCF 1σ')
    if bestfit_curve is not None:
        axs[0,0].plot(velocity_array, bestfit_curve, alpha=0.9, label='Best-fit (MAP)')
        ccf_peak = float(np.max(CCF_set))
        bf_peak = float(np.max(bestfit_curve))
        axs[0,0].axhline(ccf_peak, linestyle='--', alpha=0.8, linewidth=1, label='CCF peak')
        axs[0,0].axhline(bf_peak, linestyle='--', alpha=0.8, linewidth=1, label='Best-fit peak')
    axs[0,0].axvline(v_tot, color='r', alpha=0.4, linewidth=1.5, label='v_tot')
    axs[0,0].set_title('CCF + Signal best-fit')
    axs[0,0].legend()
    axs[0,0].set_xlabel('Velocity [km/s]')
    axs[0,0].set_xlim(velocity_range[0], velocity_range[1])

    # Top-right: priors
    axs[0,1].plot(velocity_array, prior_signal_typical, label='Signal prior')
    axs[0,1].plot(velocity_array, prior_flat_typical, label='Flat prior')
    axs[0,1].axvline(v_tot, color='r', alpha=0.4, linewidth=1.5, label='v_tot')
    axs[0,1].set_title('Prior models')
    axs[0,1].legend()
    axs[0,1].set_xlabel('Velocity [km/s]')
    axs[0,1].set_xlim(velocity_range[0], velocity_range[1])

    # Bottom-left: posterior of amplitude A
    if post_sig.size > 0:
        axs[1,0].hist(post_sig[:,1], bins=40)
        axs[1,0].set_title('Posterior of amplitude A')
        axs[1,0].set_xlabel('Amplitude A')
    else:
        axs[1,0].text(0.1, 0.5, 'No signal posterior samples', transform=axs[1,0].transAxes)

    # Bottom-right: posterior of μ with v_tot
    if post_sig.size > 0:
        axs[1,1].hist(post_sig[:,0], bins=40)
        axs[1,1].axvline(v_tot, color='r', alpha=0.4, linewidth=1.5, label='v_tot')
        axs[1,1].set_title('Posterior of mean μ')
        axs[1,1].set_xlabel('μ [km/s]')
        axs[1,1].set_xlim(velocity_range[0], velocity_range[1])
    else:
        axs[1,1].text(0.1, 0.5, 'No signal posterior samples', transform=axs[1,1].transAxes)

    # Main title with Jeffreys meaning + exposure time
    fig.suptitle(f"{main_title}", fontsize=14)
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.savefig(out_png)
    plt.close(fig)


# This function runs the MultiNest retrieval + final metrics for one planet. It is called for several planets simultaneously in the function run_parallel().
def process_planet(safe_planet):
    try:
        velocity_array = np.linspace(velocity_range[0], velocity_range[1], tot_velocity_pts)
        planet_name = safe_planet.replace('_', ' ')


        # 1) We extract the planet "v_tot" from previous petitRADTRANS simulation. It is located in a specific convenient folder
        v_tot_from_file = os.path.join(dir_results, safe_planet, f"v_tot_{safe_planet}.txt")
        try:
            v_tot = float(open(v_tot_from_file).read().strip())
        except Exception as e:
            msg = f"Warning: could not read {v_tot_from_file}: {e}"
            print(msg)
            return (planet_name, False, msg)


        # 2) For each exposure time, we look at the molecules that have CCF results
        any_molecules = False


        # 3) For each exp_time, run each molecule
        for exp_time in EXP_TIMES:
            mols = list_molecules(dir_results, safe_planet, exp_time)
            if not mols:
                print(f"{planet_name}: no molecules found for exp_time={exp_time}")
                continue
            any_molecules = True

            for molecule in mols:
                files = sorted(glob.glob(os.path.join(dir_results, safe_planet, exp_time, f"CC1d_{molecule}_*.npy")))  # This line collects all the CCF .npy files for a given planet/molecule, sorted
                if not files:
                    continue

                # I build a 2d array with all the 100 CCFs, and then calculate the inter-CCF std
                # The shape of Y is (100, 300) because we work with 100 CCFs of 300 velocity points each
                Y = np.vstack([np.load(f) for f in files])          # shape = (100, 300)
                std_ccf  = Y.std(axis=0, ddof=1)                    # shape = (300)
                sigma_scalar_eff = np.median(std_ccf)
                #sigma_ccf_scalar is the median std across velocity bins. It means "typical 1 sigma noise in a CCF value across 100 runs", it's robust against a few problematic bins.
                # The scalar is used for prior scales (e.g. baseline, amplitude), but we still use the whole vector std_ccf in the likelihood, where it matters most to weight residuals per bin.


                # 4) I load each per-molecule kernel
                kernel_v, kernel_y = load_kernel_arrays(molecule)


                # 5) Plotting part (to comment out)
                # prior_flat_typical = np.full_like(velocity_array, 0.0)
                # A_sigma_plot = 4.0 * sigma_scalar_eff
                # prior_sig_plot = A_sigma_plot * np.interp(velocity_array - v_tot, kernel_v, kernel_y, left=0.0, right=0.0)

                # Here I create the molecule output directory
                out_dir_mol = os.path.join(dir_output, safe_planet, molecule, exp_time)
                os.makedirs(out_dir_mol, exist_ok=True)

                per_ccf_results = []
                per_ccf_lines = []


                # 6) For each CCF file. Firstly I load one CCF and create a fresh per-CCF output folder
                for idx, fpath in enumerate(files):
                    one_ccf = np.load(fpath)

                    base_dir = os.path.join(out_dir_mol, f"ccf_{idx:03d}")
                    if os.path.exists(base_dir):
                        shutil.rmtree(base_dir)
                    os.makedirs(base_dir, exist_ok=True)

                    # 7) I generate the priors here. On the inflated noise scale
                    A_sigma_eff = 4.0 * sigma_scalar_eff                                        # 4.0 is weakly-informative: typical A ~ 3.2*sigma, 95% < ~7.8*sigma
                    prior_sig_eff  = signal_priors(sigma_scalar_eff, v_tot, A_sigma_eff)
                    prior_flat_eff = flat_priors(sigma_scalar_eff, A_sigma_eff)


                    # 8) Likelihoods calculation
                    loglike_sig  = loglike_signal_function(velocity_array, one_ccf, std_ccf, v_tot, kernel_v, kernel_y)
                    loglike_flat = loglike_flat_function(velocity_array, one_ccf, std_ccf, v_tot)


                    # 9) Running MultiNest on the signal model (6 params)
                    out_sig = os.path.join(base_dir, 'signal_')
                    pymultinest.run(loglike_sig,  prior_sig_eff,  n_dims=6, outputfiles_basename=out_sig,  resume=False)


                    # 10) Running MultiNest on the flat model (3 params)
                    out_flat = os.path.join(base_dir, 'flat_')
                    pymultinest.run(loglike_flat, prior_flat_eff, n_dims=3, outputfiles_basename=out_flat, resume=False)


                    # 11) Evidence (ln Z) and its error
                    logZ_s, errZ_s = read_logZ_and_err(out_sig, 6)
                    logZ_f, errZ_f = read_logZ_and_err(out_flat, 3)
                    log10B = (logZ_s - logZ_f) / np.log(10.0)
                    # Next line is calculating the error on (ln(Z_s)-ln(Z_f)/ln(10)), assuming the two Z are independents
                    sigma_log10B = np.sqrt((errZ_s if np.isfinite(errZ_s) else 0.0)**2 + (errZ_f if np.isfinite(errZ_f) else 0.0)**2) / np.log(10.0)
                    j_interpretation = jeffreys_category(log10B)


                    # 12) Reading the posteriors from MultiNest with the post-processing class Analyzer. Then padding or raising an exception if we have less than 6 posterior parameters
                    sig_an = Analyzer(n_params=6, outputfiles_basename=out_sig)
                    post_sig = sig_an.get_equal_weighted_posterior()
                    if post_sig.size == 0:
                        params_only = np.empty((0,6))      # No samples at all
                    else:
                        if post_sig.shape[1] < 6:          # Expect at least 6 parameter columns
                            raise RuntimeError(f"Posterior has only {post_sig.shape[1]} columns, expected at least 6. Check MultiNest output at {out_sig}")
                        params_only = post_sig[:, :6]


                    # 13) Printing and storing a one-line summary for that CCF (index, log10B, error, category, S/N, inflation)
                    unc = f"{sigma_log10B:.3f}" if np.isfinite(sigma_log10B) else "?"
                    line = (f"{planet_name}/{molecule} CCF #{idx:03d} (exp {exp_time}): "f"log10B={log10B:.3f}±{unc} ({j_interpretation})")

                    print(line)
                    per_ccf_lines.append(line)
                    per_ccf_results.append((idx, log10B, sigma_log10B, j_interpretation))


                    # 14) Sanity plotting (to comment out)
                    # sanity_png = os.path.join(base_dir, f"sanity_{safe_planet}_{molecule}_{idx:03d}.png")
                    # main_title = f"{j_interpretation} | log10B={log10B:.3f} | exp_time={int(exp_time) * here_change_duration} transits"
                    # sanity_plots(velocity_array, one_ccf, std_ccf, post_sig, prior_sig_plot, prior_flat_typical, v_tot, sanity_png, main_title, kernel_v, kernel_y)


                # 15) SUMMARY file (Rho, inflation, mean log10B, mean S/N, Jeffrey's category, etc)
                total = len(per_ccf_results)
                xs   = [i   for (i, l10, err, c) in per_ccf_results]                   # Array of 100 CCF realization numbers
                ys   = [l10 for (i, l10, err, c) in per_ccf_results]                   # Array of 100 log10Bs
                errs   = [err for (i, l10, err, c) in per_ccf_results]                 # Array of 100 log10B errors
                j_interpretations = [c   for (i, l10, err, c) in per_ccf_results]      # Array of 100 Jeffrey's interpretations

                log10Bs = np.array(ys, dtype=float)
                counts  = Counter(j_interpretations)
                sigma_log10Bs = np.array(errs, dtype=float)
            
                median_sigma_log10B = float(np.nanmedian(sigma_log10Bs)) if sigma_log10Bs.size else float('nan')
                median_log10B = float(np.nanmedian(log10Bs)) if log10Bs.size else float('nan')
                median_interpretation    = jeffreys_category(median_log10B) if np.isfinite(median_log10B) else "n/a"

                report_path = os.path.join(out_dir_mol, f"jeffreys_summary_{safe_planet}_{molecule}_exp{exp_time}.txt")

                with open(report_path, 'w') as f:
                    for line in per_ccf_lines:
                        f.write(line + "\n")
                    f.write("\nSummary\n")
                    f.write(f"Planet: {planet_name}\n")
                    f.write(f"Molecule: {molecule}\n")
                    f.write(f"Exposure time: {exp_time}\n")
                    f.write(f"Median log10B: {median_log10B:.3f} ({median_interpretation})\n")
                    f.write(f"Typical BF uncertainty: median σ(log10B) = {median_sigma_log10B:.3f}\n")
                    f.write("Jeffreys category breakdown:\n")

                    for category in ['supports null','negligible','substantial','strong','very strong','decisive']:
                        count = counts.get(category, 0)
                        if total > 0:
                            perc = 100 * count / total
                        else:
                            perc = 0
                        f.write(f"{category:14s}: {count:3d}  ({perc:.1f}%)\n")

        if not any_molecules:
            msg = "No molecules found for any exposure time."
            print(f"{planet_name}: {msg}")
            return (planet_name, True, msg)

        return (planet_name, True, "done")

    # Instead of the whole script crashing when one planet fails, the error is caught, and reported for that planet. The pipeline then continues with other planets.
    except Exception as e:
        msg = f"Exception in {safe_planet}: {e}"
        print(msg)
        return (safe_planet.replace('_',' '), False, msg)


# 16) This function uses parallelism computing via the Python library concurrent.futures, to run several planet retrievals simultaneously
def run_parallel():
    planets = list_safe_planets(dir_results)
    if not planets:
        print("No planets found under 'results/'.")
        return
    
    max_workers = max(1, min(len(planets), (os.cpu_count() or 2)))  # We execute as many planets as possible, as long as it's possible with regards to the number of CPUs

    print(f"Processing {len(planets)} planet(s) with {max_workers} worker(s).")
    results = []


    # 17) This creates a pool of worker processes (separate Python interpreters), and runs process_planet() for different planets, in the background.
    # It immediately returns a Future object (futs), a placeholder for the eventual result
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_planet, p): p for p in planets}

        for fut in as_completed(futs):                                          # For each planet done
            planet = futs[fut]                                                  # "fut" is the finished Future object. "planet" is then the finished planet.
            try:
                res = fut.result()
            except Exception as e:
                res = (planet.replace('_',' '), False, f"worker crashed: {e}")  # Even with a crash, we report a fail and let the main program continuing
            results.append(res)
            if res[1]:
                print(f"{res[0]} finished successfully: {res[2]}")
            else:
                print(f"{res[0]} encountered an error: {res[2]}")


    ok = sum(1 for _,s,_ in results if s)                                       # Number of planets that finished successfully (the second element of results is the success flag, True or False)
    fail = len(results) - ok                                                    # Number of failures
    print(f"All planets done. OK: {ok}, FAIL: {fail}")

if __name__=='__main__':
    run_parallel()
