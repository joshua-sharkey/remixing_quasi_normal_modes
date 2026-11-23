"""
Josh Sharkey, Dec 2025
josh.sharkey@ligo.org
"""



import sys
import os
import copy
import bilby
import logging
import warnings
import surfinBH 
import numpy as np
import configparser
import gwpy.timeseries as gts
import matplotlib.pyplot as plt
from pyRing.waveform  import TEOBPM

sys.path.append("/path/to/pipeline")
import pipeline_utils as ut
import qnm_fisher_toolkit as qnm_ftk

# environ variable for lal 
os.environ['LAL_DATA_PATH'] = "/path/to/LAL_DATA_PATH"

# this is the way
bilby.core.utils.random.seed(42)
bilby.core.utils.logger.setLevel(logging.CRITICAL)



"""
analyses
(call each with a function)
"""

# Make .gwf frame file (with GWOSC name convention),
# INPUTs:
#   frame_location: dir where output saved
#   theta_path, theta_name: location and name of .json file wiht injection param. Also icludes bilby settings (aaveform_approx, f_ref, duration, etc)
#   ifos: list of ifos in analysis
#   psd_paths: locations + names of psd, in dict, with fields as in ifos
#   calib_path, calib_name: location and name of .json with calibration data. if None, frames are made without calibration
#   sampling_f_overwrite: if not None, use this value for the sampling frequency
#   minimum_duration: minimum duration of frame file (pyRing needs them to be so long) 
# OUTPUTS:
#   * frame file: 1 per ifo
#   * plot of raw strain for each ifo
#   * plot of band passed strain for each ifo
#   * injected signal: signal + what TEOB and Kerr 10M sees for comaprsion 
#   * if calibration data present, plots calibration alpha 
#   * signal_info.json: signal timing into and channel names - to be used to build pyRing.ini
# O4A UPGRADE: DONE
def make_frame(frame_location, theta_path, theta_name, ifos, psd_paths = None, calib_path = None, calib_name = None, sampling_f_overwrite = None, minimum_duration = 64.0):
    
    ###############
    # READ INPUTS #
    ###############
    
    # read bilby settings, injection_param, psd, can calib (if not None)
    psds = {}
    if psd_paths:
        for ifo in ifos:
            psds[ifo] = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_file(psd_paths[ifo])
    else:
        for ifo in ifos:
            psds[ifo] = None
    
    theta = ut.read_dict_from_json(theta_path, theta_name)
    
    # read calibration if avaiable
    if calib_path and calib_name:
        calibration_param = ut.read_dict_from_json(calib_path, calib_name)
    
    
    #####################################
    # LOGIC TO LENGTHEN FRAME IF NEEDED #
    #####################################
    
    if theta["duration"] < minimum_duration:
        
        old_duration = theta["duration"]
        theta["duration"] = minimum_duration
        
        # shift frame start time to keep merger near the center
        extension = minimum_duration - old_duration
        theta["start_time"] = theta["start_time"] - (0.5 * extension)
    
    
    ##############################
    # SET UP WWAVEFORM GENerator #
    ##############################
    
    # set up wf gen
    waveform_arguments = dict(
        waveform_approximant=theta["waveform_approximant"],
        reference_frequency=theta["f_ref"],
        minimum_frequency=theta["f_low"],
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=theta["duration"],
        sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
        start_time = theta["start_time"],
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
    )
    
    
    ###################################################################
    # IFOS: SETUP, INJECT, PLOT, WRITE FRAME, CALCULATE RD START TIME #
    ###################################################################
    
    # used to save channels & file names(for later)
    channels_dict = {}
    f_names_dict = {}
    
    # set up ifo with detector noise & apply calibration (if calibration parameters supplied); inject signal; make a plot for a sanity check; write to file
    ifo_names = copy.deepcopy(ifos)
    ifos = bilby.gw.detector.InterferometerList(ifos)
    for ifo in ifos:
        
        psd = psds[ifo.name]
        
        if psd:
            ifo.power_spectral_density = psd # tested - this is the right way
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        else:
            ifo.set_strain_data_from_zero_noise(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        
        # apply calibration
        if calib_path and calib_name:
            freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
            max_index = len(freq_keys)-1
            
            ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
                prefix = f"recalib_{ifo.name}",
                minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
                maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
                n_points = max_index+1
            )
            
            for i in range(0, max_index+1):
                ifo.calibration_model.params[f"amplitude_{i}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{i}"])
                ifo.calibration_model.params[f"phase_{i}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{i}"])
                
            # plot (sanity check)
            f = np.linspace(float(calibration_param[f"recalib_{ifo.name}_frequency_0"]), float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]), 1000)
            alpha = ifo.calibration_model.get_calibration_factor(f)
            fig, ax = plt.subplots(figsize=(10, 4))
            l1, = ax.plot(f, np.abs(alpha) - np.ones_like(np.abs(alpha)),  color = "purple", label = "dA")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dA, %")
            ax_twin = ax.twinx()
            l2, = ax_twin.plot(f, np.angle(alpha), color = "black", linestyle='--', label = f"d$\phi$")
            ax_twin.set_ylabel(f"d$\phi$")
            ax.set_title(f"{ifo.name} Calibration")
            ax.set_xscale("log")
            plt.legend(handles=[l1, l2])
            plt.tight_layout()
            os.makedirs(frame_location, exist_ok=True)
            plt.savefig(os.path.join(frame_location, f"{ifo.name}_calibration.png"))
            plt.close()
                    
        # inject signal
        ifo.inject_signal(waveform_generator = waveform_generator, parameters = theta)
        ht = ifo.strain_data.time_domain_strain
        t = ifo.strain_data.time_array
    
        # plot (sanity check)
        plt.figure(figsize=(10, 4))
        plt.plot(t, ht, color = "purple")
        plt.xlim((theta["geocent_time"] -3.0, theta["geocent_time"] + 1.0))
        plt.title(f"{ifo.name} Strain (Truncated)")
        plt.xlabel("Time (s)")
        plt.ylabel("Strai, h")
        os.makedirs(frame_location, exist_ok=True)
        plt.savefig(os.path.join(frame_location, f"{ifo.name}_raw_strain.png"))
        plt.close()
        
        if psd:
            hf = ifo.strain_data.frequency_domain_strain
            f = ifo.strain_data.frequency_array
            
            mask = (f > max([theta["f_low"], theta["f_ref"]])) & (f < theta["f_final"]) 
            hf_bp = np.zeros_like(hf)
            hf_bp[mask] = hf[mask]
            ht_bp = bilby.core.utils.infft(hf_bp, ifo.strain_data.sampling_frequency)
            
            plt.figure(figsize=(10, 4))
            plt.plot(t, ht_bp, color = "purple")
            plt.xlim((theta["geocent_time"] -3.0, theta["geocent_time"] + 1.0))
            plt.title(f"{ifo.name} Strain (Truncated)")
            plt.xlabel("Time (s)")
            plt.ylabel("Strai, h")
            plt.savefig(os.path.join(frame_location, f"{ifo.name}_raw_strain_bandpassed.png"))
            plt.close()
        
        
        # write to gwf frame
        channel = f"{ifo.name}:GWOSC-16KHZ_R1_STRAIN"
        ts = gts.TimeSeries(
            data=ifo.strain_data.time_domain_strain,
            times = ifo.strain_data.time_array,
            channel = channel)
        site = ifo.name[0] 
        start = int(ts.t0.value) 
        dur = int(ts.duration.value)
        f_name = f"{site}-{ifo.name}_GWOSC_16KHZ_R1-{start}-{dur}.gwf"
        ts.write(os.path.join(frame_location, f_name), format="gwf")
        
        # store channel name
        channels_dict[f"{ifo.name}_channel"] = channel
        f_names_dict[f"{ifo.name}_f_name"] = f_name
        
        
    ###########################
    # CALCULATE PYRING INPUTS #
    ###########################
        
    # get remnant mass
    # fit = surfinBH.LoadFits("NRSur7dq4Remnant")
    fit = surfinBH.LoadFits("NRSur3dq8Remnant")
    M, a = ut.get_surfinBH_remnant_M_a(theta, fit)
    M_s = ut.convert_to_s(M)
        
    # get peak time
    peak_time = ut.get_peak_strain(waveform_generator, theta)
        
    # get peak of 22 mode
    #   initialize a wf gen to make just the 22 mode
    wf_args_22 = copy.deepcopy(waveform_arguments)
    wf_args_22["mode_array"] = [[2,2]]
    wf_gen = bilby.gw.WaveformGenerator(
            duration=theta["duration"],
            sampling_frequency=sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
            start_time = theta["start_time"],
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=wf_args_22,
    ) 
    #   copy of theta set to zero inclination
    theta_zi = copy.deepcopy(theta)
    theta_zi["theta_jn"] = 0
    #   get peak times 
    peak_time_22 = ut.get_peak_strain(wf_gen, theta_zi)
        
    pyRing_dict = {"M": M,
                    "a": a,
                    "M_s": M_s,
                    "coalescence_time": theta["geocent_time"],
                    "peak_strain_time": peak_time,
                    "peak_22_strain_time": peak_time_22,
                    "ra": theta["ra"],
                    "dec": theta["dec"]}
        
    # calculate ifo offsets
    for ifo in ifos:
            pyRing_dict[f"{ifo.name}_coalescence_time"] = theta["geocent_time"] + ifo.time_delay_from_geocenter(ra=theta["ra"], dec=theta["dec"], time=theta["geocent_time"])
            pyRing_dict[f"{ifo.name}_peak_strain_time"] = peak_time + ifo.time_delay_from_geocenter(ra=theta["ra"], dec=theta["dec"], time=peak_time)
            pyRing_dict[f"{ifo.name}_peak_22_strain_time"] = peak_time_22 + ifo.time_delay_from_geocenter(ra=theta["ra"], dec=theta["dec"], time=peak_time_22)
        
    # add chanels
    pyRing_dict.update(channels_dict)
    pyRing_dict.update(f_names_dict)
        
    # store in .json
    ut.write_dict_to_json(frame_location, f"signal_info.json", pyRing_dict)
    
    
    """    
    ########################################################
    # BONUS SUB PLOT SHOWING WAVEFORM SEEN BY OTHER MODELS #
    ########################################################

        
    ifos_zn = bilby.gw.detector.InterferometerList(ifo_names)
    # zero noise ifo 
    for i, ifo in enumerate(ifos_zn):
        ifo.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"],
                start_time=theta["start_time"]
        )
            
        # apply calibration
        if calib_path and calib_name:
            freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
            max_index = len(freq_keys)-1
                
            ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
                    prefix = f"recalib_{ifo.name}",
                    minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
                    maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
                    n_points = max_index+1
            )
                
            for j in range(0, max_index+1):
                    ifo.calibration_model.params[f"amplitude_{j}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{j}"])
                    ifo.calibration_model.params[f"phase_{j}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{j}"])

        # inject signal
        ifo.inject_signal(waveform_generator = waveform_generator, parameters = theta)
        ht_zn = ifo.strain_data.time_domain_strain
        t = ifo.strain_data.time_array
        
        # masks for different models
        Kerr_mask = t >= pyRing_dict[f"{ifo.name}_peak_strain_time"] + 10 * M_s
        TEOB_mask = t >= pyRing_dict[f"{ifo.name}_peak_22_strain_time"]
        
        #h_Kerr = np.zeros_like(ht_zn)
        h_Kerr = ht_zn[Kerr_mask]
        
        #h_TEOB = np.zeros_like(ht_zn)
        h_TEOB = ht_zn[TEOB_mask]

        # plot: top = full waveform + strain wiht noise if applicable, middle = TEOB cut, bottom = Kerr 10 M
        fig, (ax_top, ax_mid, ax_bot)  = plt.subplots(nrows=3, figsize=(10, 12), sharex=True)
        
        ax_top.plot(t, ht_zn, label = "injected waveform", color = "purple")
        if psd:
            hf = ifos[i].strain_data.frequency_domain_strain
            f = ifos[i].strain_data.frequency_array
            mask = (f > 2 * max([theta["f_low"], theta["f_ref"]])) & (f < theta["f_final"]) 
            hf_bp = np.zeros_like(hf)
            hf_bp[mask] = hf[mask]
            ht_dn_bp = bilby.core.utils.infft(hf_bp, ifos[i].strain_data.sampling_frequency)
            ax_top.plot(t, ht_dn_bp, label = "injection with detecor noise, bandpassed", color = "black", alpha = 0.5)
        ax_top.set_ylabel(r"Time Domain Strain, $h$")
        ax_top.legend()
        ax_top.set_title("Injected Waveform")
        
        ax_mid.plot(t[Kerr_mask], h_Kerr, label = "Kerr 10M", color = "teal")
        ax_mid.set_ylabel(r"Time Domain Strain, $h$")
        ax_mid.legend()
    
        ax_bot.plot(t[TEOB_mask], h_TEOB, label = "TEOB", color = "steelblue")
        ax_bot.set_ylabel(r"Time Domain Strain, $h$")
        ax_bot.legend()
        ax_bot.set_xlim(theta["geocent_time"] - 20*M_s, theta["geocent_time"] + 120*M_s)
        
        plt.tight_layout()
        plt.savefig(os.path.join(frame_location, f"{ifo.name}_injected_waveform.png"))
        plt.close()
        """


# Make .gwf frame file (with GWOSC name convention),
# INPUTs:
#   frame_location: dir where output saved
#   theta_path, theta_name: location and name of .json file wiht injection param - used to calculate TEOB waveform params
#   psd_paths: locations + names of psd, in dict, with fields as in ifos
#   calib_path, calib_name: location and name of .json with calibration data. if None, frames are made without calibration
#   patch_calibration: Y/N, if Y, bypass bilby's internal calibration application and use qnm_ftk instead (reduced spectral leackage)
#   zero_frame_before_start_time: Y/N, if Y, set frame to zero before the start time manually ...
#   sampling_f_overwrite: if not None, use this value for the sampling frequency
#   minimum_duration: minimum duration of frame file (pyRing needs them to be so long) 
#   teob_modes: list of tuples for modes (in TEOB format)
# OUTPUTS:
#   * frame file: 1 per ifo
#   * plot of raw strain for each ifo
#   * plot of band passed strain for each ifo
#   * if calibration data present, plots calibration alpha 
#   * signal_info.json: signal timing into and channel names - to be used to build pyRing.ini
# O4A UPGRADE: DONE
# WARNING: DATA WRITING COMMENTED OUT WHILST RE-GENERATING PLOTS
def make_teob_frame(frame_location, theta_path, theta_name, ifos, psd_paths = None, calib_path = None, calib_name = None, patch_calibration = False, zero_frame_before_start_time = False, sampling_f_overwrite = None, minimum_duration = 64.0, teob_modes = None):
    
    ###############
    # READ INPUTS #
    ###############
    
    # read bilby settings, injection_param, psd, can calib (if not None)
    psds = {}
    if psd_paths:
        for ifo in ifos:
            psds[ifo] = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_file(psd_paths[ifo])
    else:
        for ifo in ifos:
            psds[ifo] = None
    
    theta = ut.read_dict_from_json(theta_path, theta_name)
    
    # read calibration if avaiable
    if calib_path and calib_name:
        calibration_param = ut.read_dict_from_json(calib_path, calib_name)
    
    
    #####################################
    # LOGIC TO LENGTHEN FRAME IF NEEDED #
    #####################################
    
    if theta["duration"] < minimum_duration:
        
        old_duration = theta["duration"]
        theta["duration"] = minimum_duration
        
        # shift frame start time to keep merger near the center
        extension = minimum_duration - old_duration
        theta["start_time"] = theta["start_time"] - (0.5 * extension)
    

    ###############
    # SET UP IFOS #
    ###############
    
    # set up ifo with detector noise & apply calibration (if calibration parameters supplied); inject signal; make a plot for a sanity check; write to file
    ifo_names = copy.deepcopy(ifos)
    ifos = bilby.gw.detector.InterferometerList(ifos)
    for ifo in ifos:
        
        psd = psds[ifo.name]
        
        if psd:
            ifo.power_spectral_density = psd # tested - this is the right way
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        else:
            ifo.set_strain_data_from_zero_noise(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        
        # apply calibration
        if calib_path and calib_name:
            freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
            max_index = len(freq_keys)-1
            
            ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
                prefix = f"recalib_{ifo.name}",
                minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
                maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
                n_points = max_index+1
            )
            
            for i in range(0, max_index+1):
                ifo.calibration_model.params[f"amplitude_{i}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{i}"])
                ifo.calibration_model.params[f"phase_{i}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{i}"])
                
            # plot (sanity check)
            f = np.linspace(float(calibration_param[f"recalib_{ifo.name}_frequency_0"]), float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]), 1000)
            alpha = ifo.calibration_model.get_calibration_factor(f)
            fig, ax = plt.subplots(figsize=(10, 4))
            l1, = ax.plot(f, np.abs(alpha) - np.ones_like(np.abs(alpha)),  color = "purple", label = "dA")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dA, %")
            ax_twin = ax.twinx()
            l2, = ax_twin.plot(f, np.angle(alpha), color = "black", linestyle='--', label = f"d$\phi$")
            ax_twin.set_ylabel(f"d$\phi$")
            ax.set_title(f"{ifo.name} Calibration")
            ax.set_xscale("log")
            plt.legend(handles=[l1, l2])
            plt.tight_layout()
            os.makedirs(frame_location, exist_ok=True)
            plt.savefig(os.path.join(frame_location, f"{ifo.name}_calibration.png"))
            plt.close()
    

    #######################################################
    # CONVERT PARAMETERs, GENERATE TEOB WAVEFORM & INJECT #
    #######################################################

    if teob_modes:
        modes = teob_modes
    else:
        modes  = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]
    
    phases = {mode: 0.0 for mode in modes}
    model_param = {"t0": theta["geocent_time"],
                    "m1": theta["mass_1"], 
                    "m2":theta["mass_2"], 
                    "chi1": theta["a_1"], 
                    "chi2": theta["a_2"], 
                    "phases": phases, 
                    "r": theta["luminosity_distance"], 
                    "iota": theta["theta_jn"], 
                    "phi":theta["phase"], 
                    "modes":modes,
                    "TGR_params" : {}
                    }
    model = TEOBPM(**model_param)
    
    t = ifos[0].strain_data.time_array
    polarizations = model.waveform(t)
    
    h_plus, h_cross = polarizations[3], polarizations[4]
    h_complex = h_plus - 1j * h_cross
    
    ifos = ut.patch_calibration_bounds(ifos)
    ifos = qnm_ftk.inject_ht_into_ifos(h_complex, ifos, theta)

    #del model_param["modes"], model_param["phases"]
    model_param["modes"] = f"{model_param['modes']}"
    model_param["phases"] = f"{model_param['phases']}"
    ut.write_dict_to_json(frame_location, f"injected_teob_param.json", model_param) # <-- UNCOMMENT ME! 


    ###########################
    # CALCULATE PYRING INPUTS #
    ###########################
        
    # get remnant mass
    # fit = surfinBH.LoadFits("NRSur7dq4Remnant")
    fit = surfinBH.LoadFits("NRSur3dq8Remnant")
    M, a = ut.get_surfinBH_remnant_M_a(theta, fit)
    M_s = ut.convert_to_s(M)
        
    pyRing_dict = {"M": M,
                    "a": a,
                    "M_s": M_s,
                    "coalescence_time": theta["geocent_time"],
                    "peak_22_strain_time": theta["geocent_time"],
                    "ra": theta["ra"],
                    "dec": theta["dec"]}
        
    # calculate ifo offsets
    for ifo in ifos:
            pyRing_dict[f"{ifo.name}_coalescence_time"] = theta["geocent_time"] + ifo.time_delay_from_geocenter(ra=theta["ra"], dec=theta["dec"], time=theta["geocent_time"])
            pyRing_dict[f"{ifo.name}_peak_22_strain_time"] = theta["geocent_time"] + ifo.time_delay_from_geocenter(ra=theta["ra"], dec=theta["dec"], time=theta["geocent_time"])
        

    #########################
    # PLOT & WRITE TO FRAME #
    #########################

    # calculate damping time (used to set x-scale of plots)
    _,_,_,injected_tau = ut.get_pyRing_teob_remnant_and_220_qnm(theta)

    # plot TEOB wf model output directly
    plt.figure(figsize=(10, 4))
    plt.plot(t, h_plus, color = "purple", alpha = 0.5, label = "plus")
    plt.plot(t, h_cross, color = "grey", alpha = 0.5, label = "cross")
    plt.xlim((theta["geocent_time"] - 3*injected_tau, theta["geocent_time"] + 6*injected_tau))
    plt.title("TEOB waveform")
    plt.ylabel("GW Strain, h")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(frame_location, f"model_strain.png"))
    plt.close()

    # used to save channels & file names(for later)
    channels_dict = {}
    f_names_dict = {}

    for ifo in ifos:

        # get injected td strain
        ht = ifo.strain_data.time_domain_strain
        t = ifo.strain_data.time_array
    
        # plot (sanity check)
        plt.figure(figsize=(10, 4))
        plt.plot(t, ht, color = "purple")
        plt.plot([pyRing_dict[f"{ifo.name}_peak_22_strain_time"], pyRing_dict[f"{ifo.name}_peak_22_strain_time"]], [min(ht), max(ht)], "--", color = "black", alpha = 0.5)
        plt.xlim((theta["geocent_time"] -3*injected_tau, theta["geocent_time"] + 6*injected_tau))
        plt.title(f"{ifo.name} Strain (Truncated)")
        plt.xlabel("Time (s)")
        plt.ylabel("Strai, h")
        os.makedirs(frame_location, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(frame_location, f"{ifo.name}_raw_strain.png"))
        plt.close()
        
        if psd:
            hf = ifo.strain_data.frequency_domain_strain
            f = ifo.strain_data.frequency_array
            
            mask = (f > max([theta["f_low"], theta["f_ref"]])) & (f < theta["f_final"]) 
            hf_bp = np.zeros_like(hf)
            hf_bp[mask] = hf[mask]
            ht_bp = bilby.core.utils.infft(hf_bp, ifo.strain_data.sampling_frequency)
            
            plt.figure(figsize=(10, 4))
            plt.plot(t, ht_bp, color = "purple")
            plt.xlim((theta["geocent_time"] -3*injected_tau, theta["geocent_time"] + 6*injected_tau))
            plt.title(f"{ifo.name} Strain (Truncated)")
            plt.xlabel("Time (s)")
            plt.ylabel("Strai, h")
            plt.savefig(os.path.join(frame_location, f"{ifo.name}_raw_strain_bandpassed.png"))
            plt.close()
        
        
        # write to gwf frame
        channel = f"{ifo.name}:GWOSC-16KHZ_R1_STRAIN"
        
        t_frame = ifo.strain_data.time_array
        if zero_frame_before_start_time:
            ht_frame = np.zeros_like(t_frame)
            mask = t_frame >= pyRing_dict[f"{ifo.name}_peak_22_strain_time"]
            ht_frame[mask]= ifo.strain_data.time_domain_strain[mask]
            
            # plot (sanity check)
            plt.figure(figsize=(10, 4))
            plt.plot(t_frame, ht_frame, color = "purple")
            plt.plot([pyRing_dict[f"{ifo.name}_peak_22_strain_time"], pyRing_dict[f"{ifo.name}_peak_22_strain_time"]], [min(ht), max(ht)], "--", color = "black", alpha = 0.5)
            plt.xlim((theta["geocent_time"] -3*injected_tau, theta["geocent_time"] + 6*injected_tau))
            plt.title(f"{ifo.name} Strain (Truncated)")
            plt.xlabel("Time (s)")
            plt.ylabel("Strai, h")
            os.makedirs(frame_location, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(frame_location, f"{ifo.name}_raw_strain__zeroed_before_start_time.png"))
            plt.close()
            
        else:
            ht_frame = ifo.strain_data.time_domain_strain
            
        ts = gts.TimeSeries(
            data = ht_frame,
            times = t_frame,
            channel = channel)
        site = ifo.name[0] 
        start = int(ts.t0.value) 
        dur = int(ts.duration.value)
        f_name = f"{site}-{ifo.name}_GWOSC_16KHZ_R1-{start}-{dur}.gwf"
        ts.write(os.path.join(frame_location, f_name), format="gwf") # <-- UNCOMMENT ME! 
        
        # store channel name
        channels_dict[f"{ifo.name}_channel"] = channel
        f_names_dict[f"{ifo.name}_f_name"] = f_name
        
    # add chanels ti pyRIng inputs
    pyRing_dict.update(channels_dict)
    pyRing_dict.update(f_names_dict)
        
    # store pyRIng inouts in .json
    ut.write_dict_to_json(frame_location, f"signal_info.json", pyRing_dict) # <-- UNCOMMENT ME! 



# St up pyRing run
# INTPUTS:
#   * pyRing_inputs_location: dir were frame file & signal_ingo.json located (i.e. output of pipeline.make_frame)
#   * ifos: list of ifo names
#   * psd_paths: dict of pasd paths corresponding to ifos
#   * template_config: config file to base run off
#   * condor_settings_location & ..._name: .json file wiht all condor settings (exept .out .log .err and arguments)
#   * run_dir_path: path to location of outputs (i.e. output dir)
#   * run_name: base name used to name other files
#   * cp_overwrites: double dict of arguments to pass to config parser
# OUTPUTS:
#   * setps up pyRIng run
#   * returns path to .sub file
# O4A UPGRADE: DONE
def set_up_pyRing_runs(pyRing_inputs_location, ifos, psd_paths, template_config_path, condor_settings_location, condor_settings_name, run_dir_path, run_name, cp_overwrites = None):
    
    ##############################
    # READ PYRING INPUTS & SETUP #
    ##############################
    pyRing_inputs = ut.read_dict_from_json(pyRing_inputs_location, "signal_info.json")
    condor_settings = ut.read_dict_from_json(condor_settings_location, condor_settings_name)
    
    #  make run dir
    os.makedirs(run_dir_path, exist_ok=True)
    
    # initialize config parser & read template config
    cp = configparser.ConfigParser()
    cp.read(template_config_path)
    
    
    ################################
    # BUILD NEW CONFIG.INI & WRITE # 
    ################################
    
    # ifo specific arguments ...
    for ifo in ifos:
        
        ifo_lower = ifo.lower()
        
        # path to frame file
        cp["input"][f"data-{ifo_lower}"] = os.path.join(pyRing_inputs_location, pyRing_inputs[f"{ifo}_f_name"])
        
        # channel names
        cp["input"][f"channel-{ifo_lower}"] = pyRing_inputs[f"{ifo}_channel"]
        
        # if psd, set it
        if psd_paths:
             cp["input"][f"psd-{ifo_lower}"] = psd_paths[ifo]
    
    # detecors in analysis      
    cp["input"]["detectors"] = ",".join(sorted(ifos, key=str.lower))   
            
    # output path = run dir
    cp["input"]["output"] = run_dir_path
    
    # priors
    cp["Priors"]["fix-ra"] = str(pyRing_inputs["ra"])
    cp["Priors"]["fix-dec"] = str(pyRing_inputs["dec"])
    cp["Priors"]["mf-time-prior"] = str(pyRing_inputs["M"])
    
    # selection based on model
    if cp["input"]["template"] == "TEOBResumSPM":
        # TEOB runs
        cp["input"]["trigtime"] = str(pyRing_inputs["H1_peak_22_strain_time"])
        cp["Priors"]["fix-t"] = str(0.0)
    else: 
        # Kerr runs; assume 10M
        cp["input"]["trigtime"] = str(pyRing_inputs["H1_peak_strain_time"])
        cp["Priors"]["fix-t"] = str(10 * pyRing_inputs["M_s"])
    
    # add overwrites if present 
    if cp_overwrites:
        for sec, sec_dict in cp_overwrites.items():
            for field, val in sec_dict.items():
                cp[sec][field] = str(val)
    
    # write config
    run_config_path = os.path.join(run_dir_path, f"{run_name}_config.ini")
    with open(run_config_path, "w") as f:
        cp.write(f)
    
    
    ####################
    # BUILD CONDOR.SUB #
    ####################
    
    # set condor files
    condor_settings["output"] = os.path.join(run_dir_path, "condor_output.out")
    condor_settings["log"] = os.path.join(run_dir_path, "condor_log.log")
    condor_settings["error"] = os.path.join(run_dir_path, "condor_error.err")
    
    # path to condig
    condor_settings["arguments"] = f"--config {run_config_path}"
    
    # write settings
    sub_file_lines = []
    for setting, value in condor_settings.items():
        sub_file_lines.append(f"{setting} = {value}\n")
    
    # queue at end to satisfy the Condor Gods. Praise be to the Condor Gods!
    sub_file_lines.append("queue\n")
    
    run_sub_path = os.path.join(run_dir_path, f"{run_name}_condor.sub")
    with open(run_sub_path, "w") as f:
        f.writelines(sub_file_lines)
    
    return run_sub_path



# run Fisher & SNR calculations
# INPUTS:
#   * output_location: dir for outputs
#   * theta_path & ..._name: path and name of .json with injection_parameters
#   * ifo_names: list of ifos in analysis
#   * psd_paths: dict of paths to psds, structured according to above list
#   * calib_path & ..._name: ppath and name of .json with calibration data
#   * calc_type: str, "qnm" or "teob" or "teob_reparam" (for semi analytic QNM calc or numerical calc with TEOB wf - TEOB calc can be re-paramed)
#   * svd_option: if numerical TEOB calc, controls operarations to regularise inverse fisher matrix (None, "shelf", or int - do nothing, remove eigenvalues after shelf, or remove smallest N values)
#   * var_names: list of variables to incldue in calculation
#   * manual_initial_values: dict, overwrite internal calculations for initial vaules 
#   * duration_overwrite: overwirte duration of frame
#   * sampling_f_overwrite: overwrite sampling freq
# O4A UPGRADE: COMPLETE
def run_qnm_fisher_systematic_offset_calculation(output_location, theta_path, theta_name, ifo_names, psd_paths, calib_path, calib_name, calc_type = "qnm", svd_option = None, var_names = ["amp", "phase", "freq", "damp"], manual_initial_values = None, duration_overwrite = None, sampling_f_overwrite = None):

    ###############
    # READ INPUTS #
    ###############
    
    # read bilby injection_param, and, can calib 
    theta = ut.read_dict_from_json(theta_path, theta_name)
    calibration_param = ut.read_dict_from_json(calib_path, calib_name)

    # read psds
    psds = {}
    for ifo in ifo_names:
        psds[ifo] = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_file(psd_paths[ifo])


    ####################
    # THETA OVERWRITES #
    ####################

    if duration_overwrite:
        theta["duration"] = duration_overwrite
        theta["start_time"] = theta["geocent_time"] - (duration_overwrite/2.0)

    if sampling_f_overwrite:
        theta["sampling_frequency"] = sampling_f_overwrite

    ###############
    # SET UP IFOS #
    ###############
    
    ifos_dn_c = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_dn_c:
        
        ifo.power_spectral_density = psds[ifo.name] 
        ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=  theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )

        freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
        max_index = len(freq_keys)-1
            
        ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
            prefix = f"recalib_{ifo.name}",
            minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
            maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
            n_points = max_index+1
        )
            
        for i in range(0, max_index+1):
            ifo.calibration_model.params[f"amplitude_{i}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{i}"])
            ifo.calibration_model.params[f"phase_{i}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{i}"])

        # plot (sanity check)
        f = np.linspace(float(calibration_param[f"recalib_{ifo.name}_frequency_0"]), float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]), 1000)
        alpha = ifo.calibration_model.get_calibration_factor(f)
        fig, ax = plt.subplots(figsize=(10, 4))
        l1, = ax.plot(f, np.abs(alpha) - np.ones_like(np.abs(alpha)), color = "purple", label = "dA")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dA, %")
        ax_twin = ax.twinx()
        l2, = ax_twin.plot(f, np.angle(alpha), color = "black", linestyle='--', label = f"d$\phi$")
        ax_twin.set_ylabel(f"d$\phi$")
        ax.set_title(f"{ifo.name} Calibration")
        ax.set_xscale("log")
        plt.legend(handles=[l1, l2])
        plt.tight_layout()
        os.makedirs(output_location, exist_ok=True)
        plt.savefig(os.path.join(output_location, f"{ifo.name}_calibration.png"))
        plt.close()
    
    ifos_zn = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_zn:
        
        ifo.set_strain_data_from_zero_noise(
                sampling_frequency=theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
    
    ##################
    # INITIAL VALUES #
    ##################

    if (calc_type == "teob") or (calc_type == "teob_reparam"):
         initial_values = {"start": theta["geocent_time"],
                        "eps": 1e-5,
                        "m1": theta["mass_1"], 
                        "m2":theta["mass_2"], 
                        "chi1": theta["a_1"], 
                        "chi2": theta["a_2"], 
                        "r": theta["luminosity_distance"], 
                        "iota": theta["theta_jn"], 
                        "phi":theta["phase"], 
                        "phase_22":0.0,
                        "phase_21":0.0,
                        "phase_33":0.0,
                        "phase_44":0.0,
                        "phase_55":0.0,
                        }

    else:
        # get original remnant mass
        fit = surfinBH.LoadFits("NRSur3dq8Remnant")
        true_M, true_a = ut.get_surfinBH_remnant_M_a(theta, fit)
        true_f220, true_tau220 = ut.get_pyRing_qnm_fit(true_M, true_a)
        true_M_s = ut.convert_to_s(true_M)

        # set initail values
        initial_values = {"amp": 1e-21,
                        "phase": 0.0,
                        "freq": true_f220,
                        "damp": true_tau220 ,
                        "start": theta["geocent_time"],
                        "tukey": 2*true_tau220}
                  
    # overwrite if values passed
    if manual_initial_values is not None:
        for key in list(manual_initial_values.keys()):
            initial_values[key] = manual_initial_values[key]


    ######################
    # FISHER CALCULATION #
    ######################

    del_theta_sys = qnm_ftk.calculate_220_offset(ifos_zn = ifos_zn, ifos_dn = ifos_dn_c, vars = var_names, initial_values = initial_values, theta = theta, type = calc_type, svd_option = svd_option)
    
    del_theta_sys__complex_safe = {}
    for key, val in del_theta_sys.items():
        if isinstance(val, complex):
            del_theta_sys__complex_safe[f"{key}__real"] = val.real
            del_theta_sys__complex_safe[f"{key}__imag"] = val.imag
        else:
            del_theta_sys__complex_safe[key] = val

    ut.write_dict_to_json(os.path.join(output_location, "systematic_offset"), "qnm_systematic_offset.json", del_theta_sys__complex_safe)
    ut.write_dict_to_json(os.path.join(output_location, "systematic_offset"), "initial_values.json", initial_values)


    ################################
    # RECOVER WAVEFORM WITH OFFSET #
    ################################

    if calc_type == "teob":
        
        offset_vales = copy.deepcopy(initial_values)
        for key in list(del_theta_sys.keys()):
            offset_vales[key] = offset_vales[key] + del_theta_sys[key]
        
        initial_qnm = qnm_ftk.TEOB(**initial_values)
        offset_qnm = qnm_ftk.TEOB(**offset_vales)
        
        # also take the opportunity to calculate shift to 220 QNM
        _, _, true_f220, true_tau220 = ut.get_pyRing_teob_remnant_and_220_qnm(initial_values, teob_convention = True)
        _, _, f220_offset, tau220_offset = ut.get_pyRing_teob_remnant_and_220_qnm(offset_vales, teob_convention = True)
        qnm_offset = {"freq": f220_offset - true_f220,
                      "damp": tau220_offset - true_tau220,
                      }
        ut.write_dict_to_json(os.path.join(output_location, "systematic_offset"), "220_qnm_systematic_offset.json", qnm_offset)

    elif calc_type == "teob_reparam":
        
        initial_qnm = qnm_ftk.TEOB_reparam(**initial_values)
        
        # offset param in REPARAM BASIS
        offset_reparam_vals = dict(initial_qnm.params)
        for key in del_theta_sys:
            offset_reparam_vals[key] = offset_reparam_vals[key] + del_theta_sys[key]
        
        # convert back to original basis
        m1_off, m2_off, chi1_off, chi2_off = initial_qnm.reparam_to_original(offset_reparam_vals["Mf"], offset_reparam_vals["af"], offset_reparam_vals["q"], offset_reparam_vals["chi_diff"], guess_m2 = initial_qnm.m2_0, guess_chi2 = initial_qnm.chi2_0)
        r_off = np.exp(offset_reparam_vals["log_r"])
        
        # offste values in original basis
        offset_vales = copy.deepcopy(initial_values)
        offset_vales["m1"], offset_vales["m2"] = m1_off, m2_off
        offset_vales["chi1"], offset_vales["chi2"] = chi1_off, chi2_off
        offset_vales["r"] = r_off
        
        # any other vars that are shared between both bases 
        for key in del_theta_sys:
            if key not in ["Mf", "af", "q", "chi_diff", "log_r"]:
                offset_vales[key] = offset_vales[key] + del_theta_sys[key]
        
        offset_qnm = qnm_ftk.TEOB_reparam(**offset_vales)
        
        # also take the opportunity to calculate shift to 220 QNM
        _, _, true_f220, true_tau220 = ut.get_pyRing_teob_remnant_and_220_qnm(initial_values, teob_convention = True)
        _, _, f220_offset, tau220_offset = ut.get_pyRing_teob_remnant_and_220_qnm(offset_vales, teob_convention = True)
        qnm_offset = {"freq": f220_offset - true_f220,
                    "damp": tau220_offset - true_tau220,
                    }
        ut.write_dict_to_json(os.path.join(output_location, "systematic_offset"), "220_qnm_systematic_offset.json", qnm_offset)
        
    else:

        offset_vales = copy.deepcopy(initial_values)
        for key in list(del_theta_sys.keys()):
            offset_vales[key] = offset_vales[key] + del_theta_sys[key]
            
        initial_qnm = qnm_ftk.QNM(**initial_values)
        offset_qnm = qnm_ftk.QNM(**offset_vales)
    
    t = ifos_zn[0].time_array
    initial_h = qnm_ftk.call_qnm(initial_qnm, "h", t)
    offset_h = qnm_ftk.call_qnm(offset_qnm, "h", t)
    dh_sys = offset_h - initial_h


    #######################
    # DIAGNOSTIC QNM PLOT #
    #######################
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.real(initial_h), color = "teal", alpha = 0.5, label = "original waveform")
    plt.plot(t, np.real(offset_h), color = "purple", alpha = 0.5, label = "offset waveform")
    plt.plot(t, np.real(dh_sys), color = "crimson", alpha = 0.5, label = "difference")
    plt.xlim((theta["geocent_time"] - 3*true_tau220, theta["geocent_time"] + 8*true_tau220))
    plt.legend(loc="upper right")
    plt.ylabel("Strain, h")
    plt.xlabel("Time (s)")
    plt.title(f"QNM Systematic Offset")
    plt.tight_layout()
    os.makedirs(output_location, exist_ok=True)
    plt.savefig(os.path.join(output_location, f"qnm_systematic_offset.png"))
    plt.close()
    
    
    ###########################
    # INJECT INTO IFOS & PLOT #
    ###########################
    
    ifos_i = qnm_ftk.inject_ht_into_ifos(initial_h, ifos_zn, theta)
    ifos_o = qnm_ftk.inject_ht_into_ifos(offset_h, ifos_zn, theta)
    
    for n in range(0,len(ifos_zn)):
        
        h_i = ifos_i[n].time_domain_strain
        h_o = ifos_o[n].time_domain_strain
        dh_sys = h_o - h_i
        
        h_i_f = ifos_i[n].frequency_domain_strain
        h_i_f_c = qnm_ftk.apply_calibration(h_i_f, ifos_dn_c[n])
        h_c = bilby.core.utils.series.infft(h_i_f_c, ifos_dn_c[n].strain_data.sampling_frequency)
        dh_c = h_c - h_i
        
        # full plot
        plt.figure(figsize=(10, 4))
        plt.plot(t, h_i, color = "black", alpha = 0.5, label = "original waveform")
        plt.plot(t, h_o, color = "purple", alpha = 0.5,label = "offset waveform")
        plt.plot(t, dh_sys, color = "crimson", alpha = 0.5,label = "fisher difference")
        plt.plot(t, h_c, color = "forestgreen", alpha = 0.5,label = "de-calibrated waveform")
        plt.plot(t, dh_c, color = "orange", alpha = 0.5,label = "calibration difference")
        plt.xlim((theta["geocent_time"] - 3*true_tau220, theta["geocent_time"] + 8*true_tau220))
        plt.legend(loc="upper right")
        plt.ylabel("Strain, h")
        plt.xlabel("Time (s)")
        plt.title(f"QNM Systematic Offset: {ifos_zn[n].name}")
        plt.tight_layout()
        os.makedirs(output_location, exist_ok=True)
        plt.savefig(os.path.join(output_location, f"qnm_systematic_offset_{ifos_zn[n].name}.png"))
        plt.close()
        
        # above, with strains removed
        plt.figure(figsize=(10, 4))
        #plt.plot(t, h_i, color = "teal", label = "original waveform")
        #plt.plot(t, h_o, color = "purple", label = "offset waveform")
        plt.plot(t, dh_sys, color = "crimson", alpha = 0.5, label = "fisher prediction")
        #plt.plot(t, h_c, color = "forestgreen", label = "de-calibrated waveform")
        plt.plot(t, dh_c, color = "orange", alpha = 0.5, label = "calibration waveform difference")
        plt.xlim((theta["geocent_time"] - 3*true_tau220, theta["geocent_time"] + 8*true_tau220))
        plt.legend(loc="upper right")
        plt.ylabel("Strain, h")
        plt.xlabel("Time (s)")
        plt.title(f"QNM Systematic Offset: {ifos_zn[n].name}")
        plt.tight_layout()
        os.makedirs(output_location, exist_ok=True)
        plt.savefig(os.path.join(output_location, f"qnm_systematic_offset__no_strain_{ifos_zn[n].name}.png"))
        plt.close()
            
     

# numerically integrate "anstatz" for QNM shift 
# INPUTS:
#   * output_location: dir for outputs
#   * theta_path & ..._name: path and name of .json with injection_parameters; includes all ncessary bilby settings
#   * ifos: list of ifos to include in analysis
#   * psd_paths: correspondlingly structured dict of psd paths
#   * calib_path & ..._name: path and name of .json with calibration data
#   * manual_initial_values:dict of manual initial value overwrites (as in run_qnm_fisher_systematic_offset_calculation)
#   * duration_overwrite: overwrite duration?
#   * sampling_f_overwirte: overwrite sampling frequency?
# PROCESS:
#   * noise-weighted signal inner-product
#   * import functionality from qnm_ftk, e.g. to generate a QNM
# OUTPUTs:
#   * QNM properties
#   * Predicted QNM shift from anstatz 
# O4A UPGRADE: PENDING
def qnm_shift_from_ansatz(output_location, theta_path, theta_name, ifo_names, psd_paths, calib_path, calib_name, manual_initial_values = None, duration_overwrite = None, sampling_f_overwrite = None):

    ###############
    # READ INPUTS #
    ###############
    
    theta = ut.read_dict_from_json(theta_path, theta_name)
    calibration_param = ut.read_dict_from_json(calib_path, calib_name)

    # read psds
    psds = {}
    for ifo in ifo_names:
        psds[ifo] = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_file(psd_paths[ifo])

    ####################
    # THETA OVERWRITES #
    ####################

    if duration_overwrite:
        theta["duration"] = duration_overwrite
        theta["start_time"] = theta["geocent_time"] - (duration_overwrite/2.0)

    if sampling_f_overwrite:
        theta["sampling_frequency"] = sampling_f_overwrite

    ###############
    # SET UP IFOS #
    ###############
    
    ifos_dn_c = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_dn_c:
        
        ifo.power_spectral_density = psds[ifo.name] 
        ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=  theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        
        freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
        max_index = len(freq_keys)-1
            
        ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
            prefix = f"recalib_{ifo.name}",
            minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
            maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
            n_points = max_index+1
        )
            
        for i in range(0, max_index+1):
            ifo.calibration_model.params[f"amplitude_{i}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{i}"])
            ifo.calibration_model.params[f"phase_{i}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{i}"])

        # plot (sanity check)
        f = np.linspace(float(calibration_param[f"recalib_{ifo.name}_frequency_0"]), float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]), 1000)
        alpha = ifo.calibration_model.get_calibration_factor(f)
        fig, ax = plt.subplots(figsize=(10, 4))
        l1, = ax.plot(f, np.abs(alpha) - np.ones_like(np.abs(alpha)), color = "purple", label = "dA")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dA, %")
        ax_twin = ax.twinx()
        l2, = ax_twin.plot(f, np.angle(alpha), color = "black", linestyle='--', label = f"d$\phi$")
        ax_twin.set_ylabel(f"d$\phi$")
        ax.set_title(f"{ifo.name} Calibration")
        ax.set_xscale("log")
        plt.legend(handles=[l1, l2])
        plt.tight_layout()
        os.makedirs(output_location, exist_ok=True)
        plt.savefig(os.path.join(output_location, f"{ifo.name}_calibration.png"))
        plt.close()
    
    ifos_zn = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_zn:
        
        ifo.set_strain_data_from_zero_noise(
                sampling_frequency=theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )

    ##################
    # SET UP QNM OBJ #
    ##################

    # get original remnant mass
    fit = surfinBH.LoadFits("NRSur3dq8Remnant")
    true_M, true_a = ut.get_surfinBH_remnant_M_a(theta, fit)
    true_f220, true_tau220 = ut.get_pyRing_qnm_fit(true_M, true_a)
    true_M_s = ut.convert_to_s(true_M)

    # set initail values
    initial_values = {"amp": 1e-21,
                      "phase": 0.0,
                      "freq": true_f220,
                      "damp": true_tau220 ,
                      "start": theta["geocent_time"],
                      "tukey": 2*true_tau220}
                  
    # overwrite if values passed
    if manual_initial_values is not None:
        for key in list(manual_initial_values.keys()):
            initial_values[key] = manual_initial_values[key]

    qnm = qnm_ftk.QNM(**initial_values)
    
    
    ##############################################################
    # RECOVER F-DOMAIN STRAIN & CALIBRATION SPLINE; APPLY ANSATZ #
    ##############################################################

    h = qnm_ftk.call_qnm(qnm, "h", ifos_zn[0].time_array)
    ifos_zn = qnm_ftk.inject_ht_into_ifos(h, ifos_zn, theta)
    
    del_f = 0
    del_tau = 0
    norm = 0
    
    for n in range(0, len(ifo_names)):
        
        h_f = ifos_zn[n].frequency_domain_strain
        f = ifos_zn[n].frequency_array
    
        # calibration alpha
        alpha = np.ones_like(f, dtype=complex)
        f_min = ifos_dn_c[n].calibration_model.minimum_frequency
        f_max = ifos_dn_c[n].calibration_model.maximum_frequency
        mask = ifos_dn_c[n].frequency_mask & (ifos_dn_c[n].frequency_array <= f_max) & (ifos_dn_c[n].frequency_array >= f_min)
        alpha[mask] = ifos_dn_c[n].calibration_model.get_calibration_factor(f[mask])
        
        # dA and dphi splines from alpha
        dA = np.abs(alpha) - 1.0 
        dphi = np.angle(alpha)
        
        # apply ansatz - -1 * 2026 APR
        df_f = ( (f - initial_values["freq"]) * dA ) + (dphi / (2 * np.pi * initial_values["damp"]))
        dtau_f = (-2 * np.pi * (initial_values["damp"]**2) * (f - initial_values["freq"]) * dphi) + (initial_values["damp"] * dA)
        
        # inner prods
        del_f +=  bilby.gw.utils.noise_weighted_inner_product(h_f, h_f * df_f, ifos_dn_c[n].power_spectral_density_array, ifos_dn_c[n].strain_data.duration).real
        del_tau += bilby.gw.utils.noise_weighted_inner_product(h_f, h_f * dtau_f, ifos_dn_c[n].power_spectral_density_array, ifos_dn_c[n].strain_data.duration).real
        norm += bilby.gw.utils.noise_weighted_inner_product(h_f, h_f, ifos_dn_c[n].power_spectral_density_array, ifos_dn_c[n].strain_data.duration).real
    
    
    #######################
    # NORMALIZE AND STORE #
    #######################

    del_f = del_f / norm
    del_tau = del_tau / norm

    predicted_offset = {"freq": del_f,
                        "damp": del_tau}
    
    ut.write_dict_to_json(output_location, "initial_values.json", initial_values)
    ut.write_dict_to_json(output_location, "predicted_offset.json", predicted_offset)
    


# numerically integrate "anstatz" for QNM shift 
# INPUTS:
#   * output_location: dir for outputs
#   * theta_path & ..._name: path and name of .json with injection_parameters; includes all ncessary bilby settings
#   * ifos: list of ifos to include in analysis
#   * psd_paths: correspondlingly structured dict of psd paths
#   * calib_path & ..._name: path and name of .json with calibration data
#   * sampling_f_overwrite: "
#   * duration_overwrite: 2
#   * teob_modes: mode array
# OUTPUTs:
#   * wf miss-match from calibration
def calibration_teob_wf_missmatch_(output_location, theta_path, theta_name, ifo_names, psd_paths, calib_path, calib_name, sampling_f_overwrite = 16384, duration_overwrite = 8.0, teob_modes = None):
    
    ###############
    # READ INPUTS #
    ###############
    
    # read bilby settings, injection_param, psd, can calib (if not None)
    psds = {}
    if psd_paths:
        for ifo in ifo_names:
            psds[ifo] = bilby.gw.detector.psd.PowerSpectralDensity.from_power_spectral_density_file(psd_paths[ifo])
    else:
        for ifo in ifo_names:
            psds[ifo] = None
    
    theta = ut.read_dict_from_json(theta_path, theta_name)

    calibration_param = ut.read_dict_from_json(calib_path, calib_name)
    
    
    ###################
    # CHNAGE DURATION #
    ###################
    
    old_duration = theta["duration"]
    theta["duration"] = duration_overwrite
        
    # shift frame start time to keep merger near the center
    extension = duration_overwrite - old_duration
    theta["start_time"] = theta["start_time"] - (0.5 * extension)

    
    ###############
    # SET UP IFOS #
    ###############
    
    # set up ifo with detector noise & apply calibration (if calibration parameters supplied); inject signal; make a plot for a sanity check; write to file
    #ifo_names = copy.deepcopy(ifos)
    
    ifos_c = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_c:

        ifo.set_strain_data_from_zero_noise(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )
        
        # apply calibration
        if calib_path and calib_name:
            freq_keys = [key for key in calibration_param.keys() if f"recalib_{ifo.name}_frequency" in key]
            max_index = len(freq_keys)-1
            
            ifo.calibration_model = bilby.gw.detector.calibration.CubicSpline(
                prefix = f"recalib_{ifo.name}",
                minimum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_0"]),
                maximum_frequency = float(calibration_param[f"recalib_{ifo.name}_frequency_{max_index}"]),
                n_points = max_index+1
            )
            
            for i in range(0, max_index+1):
                ifo.calibration_model.params[f"amplitude_{i}"] = float(calibration_param[f"recalib_{ifo.name}_amplitude_{i}"])
                ifo.calibration_model.params[f"phase_{i}"] = float(calibration_param[f"recalib_{ifo.name}_phase_{i}"])
    
    ifos = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos:
        
        ifo.set_strain_data_from_zero_noise(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
        )
    
    ifos_dn = bilby.gw.detector.InterferometerList(ifo_names)
    for ifo in ifos_dn:
    
        ifo.power_spectral_density = psds[ifo.name] # tested - this is the right way
        ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency= sampling_f_overwrite if sampling_f_overwrite  is not None else theta["sampling_frequency"],
                duration=theta["duration"], 
                start_time=theta["start_time"],
            )

                

    #######################################################
    # CONVERT PARAMETERs, GENERATE TEOB WAVEFORM & INJECT #
    #######################################################

    if teob_modes:
        modes = teob_modes
    else:
        modes  = [(2,2), (2,1), (3,3), (3,2), (3,1), (4,4), (4,3), (4,2), (4,1), (5,5)]
    
    phases = {mode: 0.0 for mode in modes}
    model_param = {"t0": theta["geocent_time"],
                    "m1": theta["mass_1"], 
                    "m2":theta["mass_2"], 
                    "chi1": theta["a_1"], 
                    "chi2": theta["a_2"], 
                    "phases": phases, 
                    "r": theta["luminosity_distance"], 
                    "iota": theta["theta_jn"], 
                    "phi":theta["phase"], 
                    "modes":modes,
                    "TGR_params" : {}
                    }
    model = TEOBPM(**model_param)
    
    t = ifos[0].strain_data.time_array
    polarizations = model.waveform(t)
    
    h_plus, h_cross = polarizations[3], polarizations[4]
    h_complex = h_plus - 1j * h_cross
    
    ifos_c = ut.patch_calibration_bounds(ifos_c)
    ifos_c = qnm_ftk.inject_ht_into_ifos(h_complex, ifos_c, theta)
    
    ifos = qnm_ftk.inject_ht_into_ifos(h_complex, ifos, theta)

    model_param["modes"] = f"{model_param['modes']}"
    model_param["phases"] = f"{model_param['phases']}"
    ut.write_dict_to_json(output_location, f"injected_teob_param.json", model_param)
    
    
    #################################
    # MISSMATCH CALCULATION + STORE #
    #################################
    
    
    h_ah = 0.0
    mod_h = 0.0
    mod_ah = 0.0 
    
    for n in range(0, len(ifo_names)):
        
        plt.figure()
        plt.plot(ifos[n].frequency_array,ifos[n].frequency_domain_strain, color = "blue")
        plt.plot(ifos[n].frequency_array,ifos_c[n].frequency_domain_strain, color = "purple")
        plt.savefig(os.path.join(output_location, f"{ifo_names[n]}.png"))
        plt.close()
        
        h_ah += bilby.gw.utils.noise_weighted_inner_product(ifos[n].frequency_domain_strain, ifos_c[n].frequency_domain_strain, ifos_dn[n].power_spectral_density_array, ifos[n].strain_data.duration).real
        mod_h += bilby.gw.utils.noise_weighted_inner_product(ifos[n].frequency_domain_strain, ifos[n].frequency_domain_strain, ifos_dn[n].power_spectral_density_array, ifos[n].strain_data.duration).real
        mod_ah += bilby.gw.utils.noise_weighted_inner_product(ifos_c[n].frequency_domain_strain, ifos_c[n].frequency_domain_strain, ifos_dn[n].power_spectral_density_array, ifos[n].strain_data.duration).real

    missmatch = 1.0 - (h_ah/np.sqrt(mod_h * mod_ah))
    
    m_dict = {"abs": np.abs(missmatch),
              "arg": np.angle(missmatch)}
    
    ut.write_dict_to_json(output_location, "missmatch.json", m_dict)
    


    
