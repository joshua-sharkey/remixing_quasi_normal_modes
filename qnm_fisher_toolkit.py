"""
Sept 2026
josh.sharkey@ligo.org

Upgraded Fisher Toolskit: work with complex strains, inject into detecors, and include multipel detecors in analysis 
"""



import copy
import bilby
import numpy as np
from pyRing.waveform  import TEOBPM
from scipy.optimize import fsolve



"""
systematic error calculation
"""

# run systematic error calculation
# INPUTS:
#   * ifos_zn: bilby ifos, initialized wiht zero noise
#   * ifos_dn: bilby ifos, initialized with detecor noise + calibration
#   * vars: variables (in qnm obj) to inlcude in calculation
#   * initial_values: DICT of values to initalize QNM obj (initial_values[start] == theta[geocent_time])
#   * theta: dict of params (sky localisation, polarisation, geocent time of signal = QNM start time)
#   * type: type of calculation to do (qnm, or numerical teob, with or without re-param - "qnm", "teob", "teob_reparam" respectivly)
#   * svd_option: if numerical teob calc, svd_option dertermins the treatment applied to the inverse fisher matrix (see svd_inverse_fisher_matrix)
def calculate_220_offset(ifos_zn, ifos_dn, vars, initial_values, theta, type = "qnm", svd_option = None):
    
    assert theta["geocent_time"] == initial_values["start"]
    if type == "teob":
        obj = TEOB(**initial_values)
        print("Performing Numerical Calculation using pyRing's TEOB wf ...")
    elif type == "teob_reparam":
        obj = TEOB_reparam(**initial_values)
        print("Performing Numerical Calculation using pyRing's TEOB wf, with re-param basis ...")
    else:
        obj = QNM(**initial_values)
        
    f_mat = get_fisher_info_mat(obj, theta, ifos_zn, ifos_dn, vars)
    
    if (type == "teob") or (type == "teob_reparam"):
        f_mat_inv = svd_inverse_fisher_matrix(f_mat, svd_option = svd_option)
    else:
        try:
            f_mat_inv = np.linalg.inv(f_mat)
        except:
            print(f"Well this is awkward...\nGuess who didn't expect their matrix to be singular?\nYOU!")

    del_theta_sys = del_theta(obj, theta, ifos_zn, ifos_dn, vars, f_mat_inv)
    return del_theta_sys
   


# INPUTS:
#   * obj: QNM obj
#   * theta: dict of params (sky localisation, polarisation, geocent time of signal = QNM start time)
#   * ifos_zn: bilby ifos, initialized wiht zero noise
#   * ifos_dn: bilby ifos, initialized with detecor noise + calibration
#   * f_mat_inv: inverse Fisher information matrix
#   * vars: variable names to include in analysis (must match those in QNM obj)
# OUTPUTS:
#   del_theta: dict of systematic error due to calibration
def del_theta(obj, theta, ifos_zn, ifos_dn, vars, f_mat_inv):
    
    del_theta = {}
    print("Calculating Systematic Error...")
    
    for i in range(0, len(vars)):
        d_theta_i = 0
        
        for j in range(0, len(vars)):

            dh_j = call_qnm_derivative(obj, vars[j], ifos_zn[0].time_array)
            ifos_j = inject_ht_into_ifos(dh_j, ifos_zn, theta)
            
            h = call_qnm(obj, "h", ifos_zn[0].time_array)
            ifos_h = inject_ht_into_ifos(h, ifos_zn, theta)
            
            in_prod = 0
            
            for n in range(0, len(ifos_zn)):
                # sum inner product contributions form detecor network
                
                h_f = ifos_h[n].frequency_domain_strain
                h_f_c = apply_calibration(h_f, ifos_dn[n])
                del_h_f_c = h_f_c - h_f 
                
                in_prod += bilby.gw.utils.noise_weighted_inner_product(ifos_j[n].frequency_domain_strain,
                                                                       del_h_f_c,
                                                                       ifos_dn[n].power_spectral_density_array,
                                                                       ifos_dn[n].strain_data.duration
                                                                        ).real
            
            d_theta_i += f_mat_inv[i,j] * in_prod
        del_theta[vars[i]] = d_theta_i
    
    return del_theta
                
                

"""
Fisher infomration matrix
"""

# INPUTS:
#   * obj: QNM obj
#   * theta: dict of params (sky localisation, polarisation, geocent time of signal = QNM start time)
#   * ifos_zn: bilby ifos, initialized wiht zero noise
#   * ifos_dn: bilby ifos, initialized with detecor noise (can have calibration applied)
#   * vars: variable names to include in analysis (must match those in QNM obj)
# OUTPUTS:
#   * f_matrix: np array, f matrix
def get_fisher_info_mat(obj, theta, ifos_zn, ifos_dn, vars):
    
    f_matrix = np.full((len(vars),len(vars)), np.nan)
    print("Calculating Fisher Information Matrix...")
    
    for i in range(0, len(vars)):
        for j in range(0, len(vars)):
            
            # element f_ij 
            f_ij = 0.0
            
            dh_i = call_qnm_derivative(obj, vars[i], ifos_zn[0].time_array)
            ifos_i = inject_ht_into_ifos(dh_i, ifos_zn, theta)
            
            dh_j = call_qnm_derivative(obj, vars[j], ifos_zn[0].time_array)
            ifos_j = inject_ht_into_ifos(dh_j, ifos_zn, theta)
            
            for n in range(0, len(ifos_zn)):
                # sum over ifos for nw inner product  
                f_ij +=  bilby.gw.utils.noise_weighted_inner_product(ifos_i[n].frequency_domain_strain,
                                                                     ifos_j[n].frequency_domain_strain,
                                                                     ifos_dn[n].power_spectral_density_array,
                                                                     ifos_dn[n].strain_data.duration
                                                                     ).real
                
            f_matrix[i][j] = f_ij
            
    return f_matrix



# eigendecompose f_mat and build its inverse
# svd_option controls how many eigenvalues get discarded before inverting:
#   * None      - discard none, just invert
#   * "shelf"   - discard everything after the biggest gap in the eigenvalue spectrum
#   * an int, N - discard the smallest N eigenvalues
# INPUTS:
#   * f_mat: Fisher inof matrix (np array, square, symmetric)
#   * svd_option: None, "shelf", or an int 
# OUTPUTS:
#   * f_mat_inv: regularised inverse
def svd_inverse_fisher_matrix(f_mat, svd_option):
    
    f_mat = 0.5 * (f_mat + f_mat.T)
    
    eigvals, eigvecs = np.linalg.eigh(f_mat)
    
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n = len(eigvals)
    
    print(f"Cond of F mat: {np.linalg.cond(f_mat)}")
    
    ratios = eigvals[:-1] / eigvals[1:]
    print(f"{'rank':>4}  {'eigenvalue':>14}  {'ratio to next':>14}  {'vector':>50}")
    
    for i, e in enumerate(eigvals):
        ratio_to_next = ratios[i] if i < n - 1 else float("nan")
        vec = "[" + " ".join(f"{v:6.2f}" for v in eigvecs[:, i]) + "]"
        print(f"{i:4d}  {e:14.6g}  {ratio_to_next:14.6g}  {vec}")
    
    if svd_option is None:
        n_keep = n
        print("No SVD treatment -- keeping all eigenvalues")
    
    elif svd_option == "shelf":
        gap_dex = np.argmax(ratios)
        n_keep = gap_dex + 1
        print(f"Shelf found after rank {gap_dex} -- keeping {n_keep} of {n} eigenvalues")
    
    else:
        n_keep = n - svd_option
        print(f"Discarding smallest {svd_option} eigenvalues -- keeping {n_keep} of {n}")
    
    inv_eigvals = np.zeros_like(eigvals)
    inv_eigvals[:n_keep] = 1.0 / eigvals[:n_keep]
    
    f_mat_inv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T
    
    return f_mat_inv



"""
inject into ifos + calibration
"""

# inject complex td strain h into a copy of ifos, given parameters theta
# returns a COPY of ifos wiht the signal injected
def inject_ht_into_ifos(h, ifos, theta):
    
    ifos_local = copy.deepcopy(ifos)
    fs = ifos[0].strain_data.sampling_frequency
    
    h_plus = np.ascontiguousarray(h.real, dtype=float)
    h_cross = np.ascontiguousarray(-h.imag, dtype=float)
    
    h_plus_f, _ = bilby.core.utils.nfft(h_plus, fs)
    h_cross_f, _ = bilby.core.utils.nfft(h_cross, fs)
    
    injection_polarizations = {"plus": h_plus_f, "cross": h_cross_f}
    parameters = {"ra": theta["ra"],
                            "dec": theta["dec"],
                            "psi": theta["psi"],
                            "geocent_time": ifos[0].strain_data.start_time}
                            #"geocent_time": theta["geocent_time"]}
    
    #ifos_local.inject_signal(parameters, injection_polarizations)
    for ifo in ifos_local:
        ifo.inject_signal_from_waveform_polarizations(
            parameters=parameters, injection_polarizations=injection_polarizations
        )
    
    return ifos_local


# apply calibration curve to a frequency domain strain
# INPUTS:
#   * h_f: frequency domain strain
#   * ifo: bilby ifo, with calibration applied
# OUTPUTS:
#   * h_f_c: f-domain strain with calibration applied
def apply_calibration(h_f, ifo):
    
    alpha = np.ones_like(h_f, dtype=complex)
    
    f_min = ifo.calibration_model.minimum_frequency
    f_max = ifo.calibration_model.maximum_frequency
    mask = ifo.frequency_mask & (ifo.frequency_array <= f_max) & (ifo.frequency_array >= f_min)

    f = ifo.frequency_array[mask]
    alpha[mask] = ifo.calibration_model.get_calibration_factor(f)
    h_f_c = h_f * alpha

    return h_f_c


"""
qnm model + managment functions
"""

# get derivative of a given variable
def call_qnm_derivative(obj, variable, t):

    method = "dh_d" + variable
    return call_qnm(obj, method, t)

# call method_name in qnm obj,
# where t vector is in units of geocent time
def call_qnm(obj, method_name, t):
    
    method = getattr(obj, method_name)
    h = np.zeros_like(t, dtype=complex)
    mask = t >= obj.start # mask to only apply strain to t > start time
    h[mask] = method(t[mask])
    mask_tuk = (t >= (obj.start - obj.tukey)) & (t < obj.start) # mask to apply tukey roll off (with inverted dammped exponential)
    h[mask_tuk] = 0.5*(1 - np.cos( (np.pi * (t[mask_tuk] - (obj.start - obj.tukey)) )/ obj.tukey)) * method(t[mask_tuk]) #* np.exp(2*(t[mask_tuk] - obj.start)/obj.damp)
    return h


# QNM class
# INPUTS:
#   * amp: mode ampitude (of strain, h) at t = 0
#   * phase: mode phase at t = 0
#   * freq: qnm frequency in Hz (not rad/s) -- compatable with pyRing.waveform.QNM_fit
#   * damp: qnm damping time in s
#   * start: start time (relative to geocentre)
#   * tukey: length of tukey roll off on s
# METHODS:
#   * h(t): t-domain strain
#   * dh_d{var}: t-domain derrvative of a variable, where var = [amp, phase, freq, damp] (NOT start or tukey)
class QNM:
    
    def __init__(self, amp, phase, freq, damp, start, tukey):
        
        self.amp = amp
        self.phase = phase
        self.freq = freq
        self.damp = damp
        self.start = start 
        self.tukey = tukey 
    
    # qnm strain, given vector t
    def h(self, t):
        
        h = self.amp * np.exp( ( 1.0j * ( self.phase + (2.0 * np.pi * self.freq * (t - self.start) ) ) )  - ( (t - self.start)/self.damp ) )
        return h 
    
    # derrivative wrt param ...
    def dh_damp(self, t):
        
        #h = np.exp( ( 1.0j * ( self.phase + (2.0 * np.pi * self.freq * (t - self.start) ) ) )  - ( (t - self.start)/self.damp ) )
        h = self.h(t)/self.amp
        return h
    
    def dh_dphase(self, t):
        
        #h = self.amp * np.exp( ( 1.0j * ( self.phase + (2.0 * np.pi * self.freq * (t - self.start) ) ) )  - ( (t - self.start)/self.damp ) ) * 1.0j
        h = self.h(t) * 1.0j
        return h
    
    def dh_dfreq(self, t):
        
        #h = self.amp * np.exp( ( 1.0j * ( self.phase + (2.0 * np.pi * self.freq * (t - self.start) ) ) )  - ( (t - self.start)/self.damp ) ) * (2.0j * np.pi * (t - self.start))
        h = self.h(t) * (2.0j * np.pi * (t - self.start))
        return h
    
    def dh_ddamp(self, t):
        
        #h = self.amp * np.exp( ( 1.0j * ( self.phase + (2.0 * np.pi * self.freq * (t - self.start) ) ) )  - ( (t - self.start)/self.damp ) ) * ( (t - self.start) / (self.damp**2) )
        h = self.h(t) * ( (t - self.start) / (self.damp**2) )
        return h


# OUTDATED!  
class TEOB:
    
    def __init__(self, start, m1, m2, chi1, chi2, r, iota, phi, eps, **phase_lm):
        
        assert "phase_22" in phase_lm, "TEOB requires at least phase_22 (e.g. phase_22=0.0)"
        
        self.start = start   # only one start-time variable; call_qnm()'s window mask reads obj.start
        self.tukey = 0    # TEOB doesn't use a roll-off - always zero
        self.eps = eps
        
        # modes used in the waveform: whatever (l,m) the phase_lm kwargs specify
        self.modes = [(int(key[6]), int(key[7])) for key in phase_lm]
        
        # all differentiable parameters live in one flat dict -- intrinsic params + phases
        self.params = {"m1": m1, "m2": m2, "chi1": chi1, "chi2": chi2, "r": r, "iota": iota, "phi": phi}
        self.params.update(phase_lm)
    
    # t-domain complex strain, evaluated at a given params dict (defaults to self.params)
    def get_h(self, t, params = None):
        
        if params is None:
            params = self.params
        
        phases = {mode: params[f"phase_{mode[0]}{mode[1]}"] for mode in self.modes}
        model_param = {"t0": self.start,
                        "m1": params["m1"],
                        "m2": params["m2"],
                        "chi1": params["chi1"],
                        "chi2": params["chi2"],
                        "phases": phases,
                        "r": params["r"],
                        "iota": params["iota"],
                        "phi": params["phi"],
                        "modes": self.modes,
                        "TGR_params": {}
                        }
        
        model = TEOBPM(**model_param)
        polarizations = model.waveform(t)
        h_plus, h_cross = polarizations[3], polarizations[4]
        
        return h_plus - 1j*h_cross
    
    def h(self, t):
        return self.get_h(t)
    
    # numerical derivative of h wrt a single param in self.params
    # INPUTS:
    #   * t: time array
    #   * param_key: key in self.params to differentiate wrt
    # OUTPUTS:
    #   * dh_dparam: t-domain numerical derivative
    def numerical_derivative(self, t, param_key):
    
        x0 = self.params[param_key]
        
        if (x0 == 0) or (param_key == "start"):
            # fractional step undefined at zero -- use an absolute step instead
            dx = self.eps
        else:
            dx = self.eps * x0
        
        # 5-point central stencil: 4th-order accurate
        params_m2 = dict(self.params); params_m2[param_key] = x0 - 2*dx
        params_m1 = dict(self.params); params_m1[param_key] = x0 - dx
        params_p1 = dict(self.params); params_p1[param_key] = x0 + dx
        params_p2 = dict(self.params); params_p2[param_key] = x0 + 2*dx
        
        h_m2 = self.get_h(t, params_m2)
        h_m1 = self.get_h(t, params_m1)
        h_p1 = self.get_h(t, params_p1)
        h_p2 = self.get_h(t, params_p2)
        
        return (-h_p2 + 8*h_p1 - 8*h_m1 + h_m2) / (12 * dx)
    
    # hard code dh_dr to reduce numerical noise
    def dh_dr(self, t):
        return -self.get_h(t) / self.params["r"]
    
    # dynamically dispatches dh_d{var} to numerical_derivative(t, var), for any var in
    # self.params, so call_qnm_derivative(obj, var, t) works unchanged for a TEOB instance
    def __getattr__(self, name):
        if name.startswith("dh_d"):
            param_key = name[4:]
            if param_key in self.params:
                return lambda t: self.numerical_derivative(t, param_key)
        raise AttributeError(f"'TEOB' object has no attribute '{name}'")
    

class TEOB_reparam:
    
    def __init__(self, start, m1, m2, chi1, chi2, r, iota, phi, eps, **phase_lm):
        
        assert "phase_22" in phase_lm, "TEOB_reparam requires at least phase_22 (e.g. phase_22=0.0)"
        
        self.start = start  
        self.tukey = 0    
        self.eps = eps
        
        # modes used in the waveform: whatever (l,m) the phase_lm kwargs specify
        self.modes = [(int(key[6]), int(key[7])) for key in phase_lm]
        
        # fiducial (m2, chi2) - for for re-param
        self.m2_0 = m2
        self.chi2_0 = chi2
        
        # all differentiable parameters live in one flat dict, now in the REPARAM basis
        Mf, af, q, chi_diff = self.original_to_reparam(m1, m2, chi1, chi2)
        self.params = {"Mf": Mf, "af": af, "q": q, "chi_diff": chi_diff, "log_r": np.log(r), "iota": iota, "phi": phi}
        self.params.update(phase_lm)
    
    # convert (m1, m2, chi1, chi2) -> (Mf, af, q, chi_diff)
    #   * Mf, af come straight off a throwaway TEOBPM instance 
    #   * q, chi_diff are just algebra on the inputs
    @staticmethod
    def original_to_reparam(m1, m2, chi1, chi2):
        
        dummy = TEOBPM(t0 = 0.0, m1 = m1, m2 = m2, chi1 = chi1, chi2 = chi2,
                        phases = {}, r = 1.0, iota = 0.0, phi = 0.0, modes = [], TGR_params = {})
        
        Mf = dummy.Mf
        af = dummy.af
        q = m1 / m2
        chi_diff = chi1 - chi2
        
        return Mf, af, q, chi_diff
    
    # invert original_to_reparam
    # HOW:
    #   * q and chi_diff give m1 = q*m2 and chi1 = chi2 + chi_diff directly
    #   * Mf and af are nonlinear NR-fit functions - solve for the remaining 2 unknowns
    # INPUTS:
    #   * Mf, af, q, chi_diff: target values in the reparam basis
    #   * guess_m2, guess_chi2: starting guess for the solve
    @staticmethod
    def reparam_to_original(Mf, af, q, chi_diff, guess_m2, guess_chi2):
        
        def residuals(x):
            m2, chi2 = x
            m1 = q * m2
            chi1 = chi2 + chi_diff
            Mf_i, af_i, _, _ = TEOB_reparam.original_to_reparam(m1, m2, chi1, chi2)
            return [Mf_i - Mf, af_i - af]
        
        def residuals_jac(x):
            m2, chi2 = x
            dm2 = 1e-7 * m2
            dchi2 = 1e-7 if chi2 == 0 else 1e-7 * chi2
            
            r_m2_p = residuals([m2 + dm2, chi2])
            r_m2_m = residuals([m2 - dm2, chi2])
            r_chi2_p = residuals([m2, chi2 + dchi2])
            r_chi2_m = residuals([m2, chi2 - dchi2])
            
            jac = np.zeros((2, 2))
            jac[:, 0] = (np.array(r_m2_p) - np.array(r_m2_m)) / (2 * dm2)
            jac[:, 1] = (np.array(r_chi2_p) - np.array(r_chi2_m)) / (2 * dchi2)
            return jac
        
        m2, chi2 = fsolve(residuals, x0 = [guess_m2, guess_chi2], fprime = residuals_jac, xtol = 1e-13)
        m1 = q * m2
        chi1 = chi2 + chi_diff
        
        return m1, m2, chi1, chi2
    
    # t-domain complex strain - nvert param before evaulating
    def get_h(self, t, params = None):
        
        if params is None:
            params = self.params
        
        m1, m2, chi1, chi2 = self.reparam_to_original(params["Mf"], params["af"], params["q"], params["chi_diff"],
                                                        guess_m2 = self.m2_0, guess_chi2 = self.chi2_0)
        
        phases = {mode: params[f"phase_{mode[0]}{mode[1]}"] for mode in self.modes}
        model_param = {"t0": self.start,
                        "m1": m1,
                        "m2": m2,
                        "chi1": chi1,
                        "chi2": chi2,
                        "phases": phases,
                        "r": np.exp(params["log_r"]),
                        "iota": params["iota"],
                        "phi": params["phi"],
                        "modes": self.modes,
                        "TGR_params": {}
                        }
        
        model = TEOBPM(**model_param)
        polarizations = model.waveform(t)
        h_plus, h_cross = polarizations[3], polarizations[4]
        
        return h_plus - 1j*h_cross
    
    def h(self, t):
        return self.get_h(t)
    
    
    # numerical derivative of h wrt a single param in self.params
    # 5-point central stencil: 4th-order accurate
    def numerical_derivative(self, t, param_key):
    
        x0 = self.params[param_key]
        
        if (x0 == 0) or (param_key == "start"):
            # fractional step undefined at zero -- use an absolute step instead
            dx = self.eps
        else:
            dx = self.eps * x0
        
        params_m2 = dict(self.params); params_m2[param_key] = x0 - 2*dx
        params_m1 = dict(self.params); params_m1[param_key] = x0 - dx
        params_p1 = dict(self.params); params_p1[param_key] = x0 + dx
        params_p2 = dict(self.params); params_p2[param_key] = x0 + 2*dx
        
        h_m2 = self.get_h(t, params_m2)
        h_m1 = self.get_h(t, params_m1)
        h_p1 = self.get_h(t, params_p1)
        h_p2 = self.get_h(t, params_p2)
        
        return (-h_p2 + 8*h_p1 - 8*h_m1 + h_m2) / (12 * dx)
    
    # dynamically dispatches dh_d{var} to numerical_derivative(t, var), for any var in
    # self.params, so call_qnm_derivative(obj, var, t) works unchanged for a TEOB_reparam instance
    def __getattr__(self, name):
        if name.startswith("dh_d"):
            param_key = name[4:]
            if param_key in self.params:
                return lambda t: self.numerical_derivative(t, param_key)
        raise AttributeError(f"'TEOB_reparam' object has no attribute '{name}'")


