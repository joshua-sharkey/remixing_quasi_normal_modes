"""
Josh Sharkey, Nov 2025
josh.sharkey@ligo.org

Utilities for pipline
"""



import bilby
import os
import json
import copy
import math
import h5py
import surfinBH
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from pyRing.waveform import QNM_fit
from pyRing.waveform  import TEOBPM
from matplotlib.collections import LineCollection
from lal import MTSUN_SI as M_sun_s, MSUN_SI as M_sun 
from astropy.cosmology import default_cosmology, z_at_value
from lalsimulation import SimInspiralTransformPrecessingWvf2PE as lal_to_bilby_spins 



"""
.json reading/writing
"""

# write dict to .json called f_name, in dir
def write_dict_to_json(dir, f_name, dict):

    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, f_name), "w") as file:
        json.dump(dict, file)
        

# reads contents of f_name in dir into .json file
def read_dict_from_json(dir, f_name):
    
    if not os.path.exists(os.path.join(dir, f_name)):
        raise FileNotFoundError(f"File '{f_name}' does not exist in '{dir}'")
    else:
        with open(os.path.join(dir, f_name), "r") as file:
            dict = json.load(file)
    
    return dict



"""
bh remnant & qnm calculations
nb: assumes bilby-style injection_parameters
"""

# for set of injection param, calculate the remant BH mass M and spin a
# INPUTS:
#   * theta: bilby injection param dict
#   * fit: surfinBH fit
#   * detecor_frame: True/False, flag for conversion to detector frame or not
# OUTPUTS:
#   * M, a: remant mass M (solar masses) and spin a (dimensionless)
def get_surfinBH_remnant_M_a(theta, fit, detector_frame=True, f_ref = 20.0):
    
    # transofmr variables to surfinBH format
    q = theta["mass_1"]/theta["mass_2"] 
    phi = convert_spin_to_lal(theta=theta, f_ref=f_ref)
    chi_a = np.array([phi["spin_1x"], phi["spin_1y"], phi["spin_1z"]])
    chi_b = np.array([phi["spin_2x"], phi["spin_2y"], phi["spin_2z"]])
 
    # calculate remant properties & rescale
    mf_surf, mf_err_surf = fit.mf(q, chi_a, chi_b)
    M_surf= mf_surf * (theta["mass_1"] + theta["mass_2"])
    chif_surf, chif_surf_err = fit.chif(q, chi_a, chi_b)
    a = np.linalg.norm(chif_surf)
    
    if not detector_frame:
        # convert from source frame to detector frame
        cosmo = default_cosmology.get()
        lum_distance = theta["luminosity_distance"] * u.Mpc
        z = z_at_value(cosmo.luminosity_distance, lum_distance)
        M = (1 + z.value) * M_surf
    else:
        M = M_surf
    
    return M, a

# get f, tau of lmn QNM
# INPUTS:
#   * M: remant mass, solar masses
#   * a: remnant spin
#   * l, m, n = qnm indicies 
# OUTPUTS:
#   f, tau = qnm frequency and damping time
# NOTE: f in Hz, not rad/s
def get_pyRing_qnm_fit(M, a, l=2, m=2, n=0):
    
    f = QNM_fit(l,m,n).f(M, a) 
    tau = QNM_fit(l,m,n).tau(M, a)
    
    return f, tau 

# get f, tau from lmn QNM, taking bilby style injection param as an input
# INPUTS:
#   * theta: bilby injection param dict
#   * fit: surfinBH fit
#   * detecor_frame: True/False, flag for conversion to detector frame or not
#   * l,m,n = qnm indicies 
# OUTPUTS:
#   * qnm frequency & damping time: f, tau
def get_pyRing_qnm_fit_via_surfinBH(theta, fit, l=2, m=2, n=0, detector_frame=True):

    M, a = get_surfinBH_remnant_M_a(theta, fit, detector_frame)
    f, tau = get_pyRing_qnm_fit(M, a, l, m, n)

    return f, tau

# get: M, a, f_220, tau_220 from pyRing TEOB, given theta
# INPUTS:
#   * theta: bilby injectin param dict (can also follow toeb convention)
# OUTPUTS:
#   Mf, af, f_220, tau_220
def get_pyRing_teob_remnant_and_220_qnm(theta, teob_convention = False):

    if teob_convention:
        m1, m2, chi1, chi2, phase_22 = theta["m1"], theta["m2"], theta["chi1"], theta["chi2"], theta["phase_22"]
    else:
        m1, m2, chi1, chi2, phase_22 = theta["mass_1"], theta["mass_2"], theta["a_1"], theta["a_2"], 0.0
    tmp = TEOBPM(0, m1, m2, chi1, chi2, {(2,2):phase_22}, 100, 0, 0, [(2,2)], {})
    Mf = tmp.JimenezFortezaRemnantMass()
    af = tmp.JimenezFortezaRemnantSpin()

    tM = tmp.Mf * M_sun_s
    f_220 = tmp.fit_coefficients[(2, 2)]['omega1'] / (2 * np.pi * tM)
    tau_220 = tM / tmp.fit_coefficients[(2, 2)]['alpha1'] 

    return Mf, af, f_220, tau_220



# convetr mass in solar masses to seconds
# INPUTS: M_sol
# OUTPUTS: converted from solar masses >>> seconds
def convert_to_s(M_sol):
    return M_sol * M_sun_s



"""
waveform calculations
"""    

# get time of peak strain
# INPUTS:
#   * wf_gen: waveform generator (ready)
#   * theta: bilby styl;e injection param
# OUTPUTS:
#   * peak_time: time of peak strain
def get_peak_strain(wf_gen, theta):
    
    # get h and t array from wf gen
    # nb - coelescence time aligned at start of array
    ht_wf = wf_gen.time_domain_strain(theta)
    t_wf = wf_gen.time_array
    
    # find time where |h| peaks, the correct back to sanity
    # shout out to the #### who thought this alignment comnvention was a good idea
    mod_ht_square = ht_wf["plus"]**2 + ht_wf["cross"]**2
    peak_index = np.argmax(mod_ht_square)
    if peak_index < len(mod_ht_square)/2:
        peak_time = theta["geocent_time"] + (t_wf[peak_index] - t_wf[0])
    else:
        peak_time = theta["geocent_time"] + (t_wf[peak_index] - t_wf[len(t_wf)-1])
    
    return peak_time



"""
parameter conversion functions

'Do not traverse that barren land scape,
Do not look back and dwell,
For only pain and suffering can be found there'
 - JS, 2025, after spending many an hours exasperated by parameter conversions
"""

# convert spins from bilby to lal sim Cartesian 
# feeds into bilby.gw.conversion.bilby_to_lalsimulation_spins
# INPUTS:
#   * theta: bilby injection param dict
#   * f_ref: reference frequency
# OUTPUT
#   * phi
def convert_spin_to_lal(theta, f_ref):

    phi = copy.deepcopy(theta)

    phi["iota"], phi["spin_1x"], phi["spin_1y"], phi["spin_1z"], phi["spin_2x"], phi["spin_2y"], phi["spin_2z"] = bilby.gw.conversion.bilby_to_lalsimulation_spins(theta["theta_jn"], theta["phi_jl"], theta["tilt_1"], theta["tilt_2"], theta["phi_12"], theta["a_1"], theta["a_2"], theta["mass_1"] * M_sun, theta["mass_2"] * M_sun, f_ref, theta["phase"])
    
    del phi["theta_jn"], phi["phi_jl"], phi["tilt_1"], phi["tilt_2"], phi["phi_12"], phi["a_1"], phi["a_2"]
    return phi

# inverse of above - convert spins from lal to bilby
# feeds into lalsiimulation.SimInspiralTransformPrecessingWvf2PE
def convert_spin_to_bilby(phi, f_ref):

    theta = copy.deepcopy(phi)

    theta["theta_jn"], theta["phi_jl"], theta["tilt_1"], theta["tilt_2"], theta["phi_12"], theta["a_1"], theta["a_2"] = lal_to_bilby_spins (incl = phi["iota"], m1 = phi["mass_1"], m2 = phi["mass_2"], S1x=phi["spin_1x"], S1y=phi["spin_1y"], S1z=phi["spin_1z"], S2x=phi["spin_2x"], S2y=phi["spin_2y"], S2z=phi["spin_2z"], fRef = f_ref, phiRef = phi["phase"])
    
    del theta["iota"], theta["spin_1x"], theta["spin_1y"], theta["spin_1z"], theta["spin_2x"], theta["spin_2y"], theta["spin_2z"]
    return theta



"""
bilby utils
"""

# patch calibration model to avoid extrapolating outside the calibration model's bounds
def patch_calibration_bounds(ifos):

    ifos_local = copy.deepcopy(ifos)

    for ifo in ifos_local:

        f_min = getattr(ifo.calibration_model, 'minimum_frequency', None)
        f_max = getattr(ifo.calibration_model, 'maximum_frequency', None)

        if f_min is None or f_max is None:
            print(f"Bypassing calibration bounds patch for {ifo.name}...")
            continue

        original_alpha = ifo.calibration_model.get_calibration_factor

        def _bounded(frequency_array, **kwargs):
            alpha = np.ones(len(frequency_array), dtype=complex)
            mask = (frequency_array >= f_min) & (frequency_array <= f_max)
            if mask.any():
                alpha[mask] = original_alpha(frequency_array[mask], **kwargs)
            return alpha

        ifo.calibration_model.get_calibration_factor = _bounded

    return ifos_local



"""
post processing pyRing posteriors
"""

# bassed on namib, see https://github.com/vascogennari/namib/blob/3d84e5e137b8bcc7f271fa36f08efd037734bbda/namib/utils.py#L303
def get_remnant_post_from_teob(post):
    
    N = len(post["m1"])
    Mf = np.zeros(N)
    af = np.zeros(N)
    
    for i in range(N):
        
        m1, m2, chi1, chi2 = post["m1"][i], post["m2"][i], post["chi1"][i], post["chi2"][i]
        tmp = TEOBPM(0, m1, m2, chi1, chi2, {}, 100, 0, 0, [], {})
        Mf[i] = tmp.JimenezFortezaRemnantMass()
        af[i] = tmp.JimenezFortezaRemnantSpin()
    
    #names = ["Mf", "af"]
    #dtypes = [(names[0], np.float64), (names[1], np.float64)]
    #Mf_af_post = np.zeros(N, dtype=dtypes)
    #Mf_af_post[names[0]] = Mf
    #Mf_af_post[names[1]] = af
    
    return Mf, af

def get_220_qnm_post_from_teob(post):
    
    N = len(post["m1"])
    f = np.zeros(N)
    tau = np.zeros(N)
    
    for i in range(N):
        
        m1, m2, chi1, chi2, phase_22 = post["m1"][i], post["m2"][i], post["chi1"][i], post["chi2"][i], post["phase_22"][i]
        tmp = TEOBPM(0, m1, m2, chi1, chi2, {(2,2):phase_22}, 100, 0, 0, [(2,2)], {})
        
        tM = tmp.Mf * M_sun_s
        f[i] = tmp.fit_coefficients[(2, 2)]['omega1'] / (2 * np.pi * tM)
        tau[i] = tM / tmp.fit_coefficients[(2, 2)]['alpha1']   
        
    return f, tau

# read pyRing posterior.dat; retunr structured numpy array & column names
def read_pyRing_post(post_path):
    
    nes_post_path = post_path.replace("posterior.dat", "result.hdf5")
    assert os.path.exists(post_path) or os.path.exists(nes_post_path), (f"NEITHER EXIST: {post_path} & {nes_post_path}")
    
    if os.path.exists(post_path):
        post = np.genfromtxt(post_path, delimiter=" ", names=True)
        names = post.dtype.names
    else:
        with h5py.File(nes_post_path, "r") as f:
            post = f["posterior_samples"][()]
            names = post.dtype.names
        
    return post, names

# take post from a Kerr 220 run: get post in f_220 and tau_220 
def get_220_qnm_post_from_kerr_run(post):
    l,m,n = 2,2,0
    
    f = [QNM_fit(l,m,n).f(mf, af) for mf, af in zip(post["Mf"], post["af"])]
    tau = [QNM_fit(l,m,n).tau(mf, af) for mf, af in zip(post["Mf"], post["af"])]
    
    names = ["f_220", "tau_220"]
    dtypes = [(names[0], np.float64), (names[1], np.float64)]
    f_tau_post = np.zeros(len(f), dtype=dtypes)
    f_tau_post[names[0]] = f
    f_tau_post[names[1]] = tau
    
    return f_tau_post, names

# take a TEOB posterior and convert it to something *actually helpful*: remnant mass, spin, and f_220 + fau_220
def get_220_qnm_post_from_teob_run(post, gr_dev = False):
    
    # function coppied from namib
    Mf, af = get_remnant_post_from_teob(post)
        
    # use pyRing to get 220 frequency + damping time
    #l,m,n = 2,2,0
    #f = [QNM_fit(l,m,n).f(m_f, a_f) for m_f, a_f in zip(Mf, af)] # <--- might be cooked
    #tau = [QNM_fit(l,m,n).tau(m_f, a_f) for m_f, a_f in zip(Mf, af)] # <--- might be cooked
    
    f, tau = get_220_qnm_post_from_teob(post)
    
    if gr_dev:
        del_f = [d * f_ for d,f_ in zip(post["domega_220"], f)]
        del_tau = [d * t for d,t in zip(post["dtau_220"], tau)]
        
    names = ["Mf", "af", "f_220", "tau_220"]
    dtypes = [(names[0], np.float64), (names[1], np.float64), (names[2], np.float64), (names[3], np.float64)]
    
    if gr_dev:
        names.extend(["del_f_220", "del_tau_220"])
        dtypes.extend([(names[4], np.float64), (names[5], np.float64)])
    
    converted_post = np.zeros(len(f), dtype=dtypes)
    converted_post[names[0]] = Mf
    converted_post[names[1]] = af
    converted_post[names[2]] = f
    converted_post[names[3]] = tau
    
    if gr_dev:
        converted_post[names[4]] = del_f
        converted_post[names[5]] = del_tau
        
    return converted_post, names


"""
SNR Contour Plots
(may be depriciated)
"""

# f/tau contour plot (for f and tau deviation by SNR)
def two_pannel_snr_comtour_plot(seg_top, seg_bottom, snr_list, qnm_f, total_mass_list, y_label_top, y_label_bottom, x_label, title, save_path):
    
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)
    
    #######
    # TOP #
    #######
    
    lc_top = LineCollection(seg_top, cmap="viridis", linewidths=2, array=snr_list)
    ax_top.add_collection(lc_top)
    ax_top.autoscale()
    fig.colorbar(lc_top, ax=ax_top, label="SNR")
    
    # add vertical lines for qnm frequencies
    
    ymin_top, ymax_top = ax_top.get_ylim()
    if np.abs(ymin_top) > np.abs(ymax_top):
        ax_top.set_ylim(1.2 * ymin_top, ymax_top)
        ylab_top = ymin_top * 1.1
    else:
        ax_top.set_ylim(ymin_top, 1.2 * ymax_top)
        ylab_top = ymax_top * 1.1
    
    for i, qnm in enumerate(qnm_f):
        mass = total_mass_list[i]
        ax_top.axvline(qnm, color='black', linestyle='--', linewidth=0.8, label='_nolegend_')
        ax_top.text(
            qnm,
            ylab_top,
            f"{mass}",
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1),
        )
    
    ax_top.set_ylabel(y_label_top)
    ax_top.set_title(title)
    
    ##########
    # BOTTOM #
    ##########

    lc_bot = LineCollection(seg_bottom, cmap="viridis", linewidths=2, array=snr_list)
    ax_bot.add_collection(lc_bot)
    ax_bot.autoscale()
    fig.colorbar(lc_bot, ax=ax_bot, label="SNR")
    
    # add vertical lines for qnm frequencies
    
    ymin_bot, ymax_bot = ax_bot.get_ylim()
    if np.abs(ymin_bot) > np.abs(ymax_bot):
        ax_bot.set_ylim(1.2 * ymin_bot, ymax_bot)
        ylab_bot = ymin_bot * 1.1
    else:
        ax_bot.set_ylim(ymin_bot, 1.2 * ymax_bot)
        ylab_bot = ymax_bot * 1.1
    
    for i, qnm in enumerate(qnm_f):
        mass = total_mass_list[i]
        ax_bot.axvline(qnm, color='black', linestyle='--', linewidth=0.8, label='_nolegend_')
        ax_bot.text(
            qnm,
            ylab_bot,
            f"{mass}",
            verticalalignment='bottom',
            horizontalalignment='center',
            fontsize=8,
            color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1),
        )
    
    ax_bot.set_xlabel("220 Mode Frequency (Hz)")
    ax_bot.set_ylabel(y_label_bottom)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
