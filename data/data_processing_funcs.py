from itertools import compress
import os
import inspect
import json

import numpy as np
import scipy.io as scio
import scipy.signal as sig
import scipy.stats as stats
from findiff import FinDiff
import pandas as pd

def json_to_dict(filepath):

    """
    Convencience function for opening and converting .json file to dict
    """

    with open(filepath, 'r') as f:
        data_dict = json.load(f) 

    return data_dict

def dict_to_json(dict, filename):
    
    """
    Convencience function for saving dict to .json file
    """

    with open(filename, 'w') as file:
        json.dump(dict, file)
    
    return

def rshape(data):
    data = np.reshape(data, data.shape[1])
    return [np.reshape(test, test.shape[0]) for test in data]

def load_RT_data_dict(raw = False, subs = None):

    '''Make a dictionary with participant ids as keys and lists of PVT reaction time data as values. 
       Lists contain arrays of reaction time sequences in order of constant routine time.'''

    module_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(load_RT_data_dict))) # get directory of this module
    data = (scio.loadmat(f"{module_dir}/pvtSD.mat"))['pvtSD'] if not raw else (scio.loadmat(f"{module_dir}/pvtSD_with_raw.mat"))['pvtSD']

    if subs is None:
        subs = np.arange(data.shape[1])
    elif isinstance(subs, int):
        subs = np.array([subs])

    RTs_dict = {data[0, i][0][0]: rshape(data[0, i][7]) for i in subs}
    return RTs_dict

def load_MT_data_dict():

    module_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(load_MT_data_dict))) # get directory of this module
    MT_data = (scio.loadmat(f"{module_dir}/time_data_converted.mat"))['data']

    ids = [id_arr[0] for id_arr in MT_data[0, 0][0][0]]
    all_times = MT_data[0, 0][1].T*24
    DLMOs = MT_data[0, 0][2][0]*24
    DLMOffs = MT_data[0, 0][3][0]*24

    times_dict = {id: {'time_arr': times, 'DLMO': DLMO, 'DLMOff': DLMOff} for id, times, DLMO, DLMOff in zip(ids, all_times, DLMOs, DLMOffs)}

    return times_dict

def load_synthetic_data_dict(filename, session_ind = None):

    rRTs_dict = json_to_dict(f'{filename}')
    if session_ind is None:
        rRTs_dict = {id: np.array(arr) for id, arr in rRTs_dict.items()}
    else:
        rRTs_dict = {id: np.array(arr)[session_ind] for id, arr in rRTs_dict.items()}

    return rRTs_dict

def load_genotype_df():

    module_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(load_genotype_df))) # get directory of this module
    genotype_df = pd.read_csv(f'{module_dir}\GWA_AGE_Genotype.csv')

    return genotype_df

def pool_individual_data_over_time(ind_data, time_ind_pairs):

    return [np.concatenate(ind_data[time_ind_pairs[0]: time_ind_pairs[1]])]

def pool_all_individual_data_over_time(RTs_dict, time_ind_pairs):

    RTs_dict_pooled = {}

    for ind_id, ind_data in RTs_dict.items():

        pooled_data = pool_individual_data_over_time(ind_data, time_ind_pairs)
        RTs_dict_pooled[ind_id] = {f'{time_ind_pair[0]} - {time_ind_pair[1] - 1}': pooled_data[i] for i, time_ind_pair in enumerate(time_ind_pairs)}

    return RTs_dict_pooled

def pool_all_data(data_dict, upper_cutoff, lower_cutoff, tpts = None):

    if tpts is None:
        dummy_key = list(data_dict.keys())[0]
        tpts = np.arange(len(data_dict[dummy_key]))
    elif isinstance(tpts, int) or isinstance(tpts, np.int32) or isinstance(tpts, np.int64):
        tpts = [tpts]

    RTs_all = np.concatenate([np.concatenate([sessions[i] for i in tpts]) for sessions in list(data_dict.values())])
    RTs_all = np.sort(RTs_all[~np.isnan(RTs_all)])
    RTs_all = RTs_all[RTs_all <= upper_cutoff]
    RTs_all = RTs_all[RTs_all >= lower_cutoff]

    return RTs_all

def group_data_by_const_rout(RTs_dict, tpt_inds = None, tpts = None):

    if tpt_inds is None:
        tpt_inds = np.arange(20)
    if tpts is None:
        tpts = 2*(np.array(tpt_inds) + 1)

    RTs_by_CR_time = []
    for t in tpt_inds:
        RTs_t = [ind_data[t] for ind_data in RTs_dict.values()]
        RTs_by_CR_time.append(np.concatenate(RTs_t))
    
    return RTs_by_CR_time, tpts

def get_session_times_by_circadian_ref(time_dict, ref = 'DLMO', as_dict = False, concatenate = True):

    if not ref in ['DLMO', 'DLMOff']:
        raise ValueError(f'"ref" argument must be "DLMO" or "DLMOff", not "{ref}"')
    
    if as_dict:
        time_from_ref_out = {ind_key: (ind_dict["time_arr"] - ind_dict[ref])[1:] for ind_key, ind_dict in time_dict.items()}
    else:
        reference_time = np.array([times[ref] for times in time_dict.values()])
        time_arrs = np.array([times['time_arr'] for times in time_dict.values()]) # time of day for each participant and PVT session
        time_arrs = time_arrs[:, 1:] # remove sstarting (wake-up) time
        time_from_ref_out = [time_arrs[ind] - reference_time[ind] for ind in range(len(time_dict.keys()))] # time of day relative to DLMO
        if concatenate:
            time_from_ref_out = np.concatenate(time_from_ref_out)

    return time_from_ref_out
    
def group_data_by_circ_ref(RTs_dict, time_dict, edges = None, time_from_ref_arr = None, ref = 'DLMO', pool = True):

    # filter out time data for participants cut from analysis
    ids = [id for id in time_dict.keys() if id in RTs_dict.keys()]
    time_dict = {part_id: times for part_id, times in time_dict.items() if part_id in ids}
    
    if time_from_ref_arr is None:
        time_from_ref_arr = get_session_times_by_circadian_ref(time_dict, ref)

    # only include participants who have available melatonin data
    RTs_list = [dict_entry for sub, dict_entry in RTs_dict.items() if sub in ids]

    sessions_list = []
    for ind_sessions in RTs_list:
        sessions_list += ind_sessions

    sessions_list = list(compress(sessions_list, ~np.isnan(time_from_ref_arr))) # remove sessions from list if timepoint is nan (all arrays are empty)
    time_from_ref_arr = time_from_ref_arr[~np.isnan(time_from_ref_arr)] # remove nan timepoints
    sessions_list = [sessions_list[i] for i in np.argsort(time_from_ref_arr)]
    time_from_ref_arr = np.sort(time_from_ref_arr)
    
    if edges is None:
        min_val = np.nanmin(time_from_ref_arr)
        max_val = np.nanmax(time_from_ref_arr)
        edges = np.arange(min_val, max_val + 1/12, 1/12)

    regrouped_sessions = []
    
    for i, (lower_edge, upper_edge) in enumerate(zip(edges[:-1], edges[1:])):

        if np.sum(np.nonzero(time_from_ref_arr >= lower_edge)[0]) == 0:
            regrouped_sessions.append([])
            continue
        if np.sum(np.nonzero(time_from_ref_arr < upper_edge)[0]) == 0:
            regrouped_sessions.append([])
            continue

        lower_ind = np.min(np.nonzero(time_from_ref_arr >= lower_edge)[0])

        if not i == len(edges) - 2:
            upper_ind = np.max(np.nonzero(time_from_ref_arr < upper_edge)[0])
        else:
            upper_ind = np.max(np.nonzero(time_from_ref_arr <= upper_edge)[0]) # make last bin inclusive of upper bound
        
        sessions_in_bin = sessions_list[lower_ind:upper_ind]
        regrouped_sessions.append(sessions_in_bin)
    
    if pool:
        regrouped_sessions = [np.concatenate(RT_sequences_t) if not RT_sequences_t == [] else np.array([]) for RT_sequences_t in regrouped_sessions]

    times = (edges[:-1] + edges[1:])/2
    return regrouped_sessions, times

def filter_data(RTs_dict, lower_cutoff, upper_cutoff, genotype = None, remove_adaptation_effect = True, remove_circ_misaligned = True):

    '''Remove anomalies and data outside desired bounds'''
    
    if not genotype is None:
        genotype_df = load_genotype_df()
        genotype_dict = {part_id: part_genotype for part_id, part_genotype in zip(genotype_df['ID'].values, genotype_df['Genotype'].values)}
        RTs_dict = {key: session_list for key, session_list in RTs_dict.items() if key in genotype_dict.keys() and genotype_dict[key] == genotype}

    if remove_circ_misaligned:
        for key in ['p00018',  'p00414',  'p00734',  'p00762',  'p01123',  'p01200']:
            if key in list(RTs_dict.keys()):
                del RTs_dict[key]

    for key in RTs_dict.keys():

        individual_session_list = RTs_dict[key]

        if remove_adaptation_effect:
            if key == 'p01342': # this participant has a string of 10 non-responses at the start of the first session
                individual_session_list[0] = individual_session_list[0][10:]
            else: # for everyone else, remove "adaptation effect" from first session
                individual_session_list[0] = individual_session_list[0][2:]

        for t, RT_sequence in enumerate(individual_session_list):
            individual_session_list[t] = RT_sequence[(RT_sequence < upper_cutoff)*(RT_sequence > lower_cutoff)]

        RTs_dict[key] = individual_session_list
    
    return RTs_dict

def get_individual_data(RTs_dict, time_inds, scale_factor):
    if not hasattr(time_inds, "__len__"):
        time_inds = np.array([time_inds])
    return {key: np.concatenate([1/sessions[i]*scale_factor for i in time_inds]) for key, sessions in RTs_dict.items()}

def get_standard_data_DLMO(lower_cutoff = 167, upper_cutoff = 10000, scale_factor = 1000, time_edges = np.arange(-15, 29, 2), genotype = None, remove_circ_misaligned = True, shift_data = False):

    '''Convenience function to load DLMO-aligned data in a standard way.'''

    RTs_dict = load_RT_data_dict(raw = True, subs = None)
    time_dict = load_MT_data_dict()

    RTs_dict = filter_data(RTs_dict, lower_cutoff, upper_cutoff, genotype = genotype, remove_circ_misaligned = remove_circ_misaligned)

    pooled_data, T_arr = group_data_by_circ_ref(RTs_dict, time_dict, edges = time_edges)
    if shift_data:
        pooled_data = [1/(p_data_t[~np.isnan(p_data_t)] - lower_cutoff)*scale_factor for p_data_t in pooled_data]
    else:
        pooled_data = [1/p_data_t[~np.isnan(p_data_t)]*scale_factor for p_data_t in pooled_data]

    return pooled_data, T_arr

def get_standard_data_CR(lower_cutoff = 167, upper_cutoff = 10000, scale_factor = 1000, remove_circ_misaligned = True):

    '''Convenience function to load CR-aligned data in a standard way.'''

    RTs_dict = load_RT_data_dict(raw = True, subs = None)

    RTs_dict = filter_data(RTs_dict, lower_cutoff = lower_cutoff, upper_cutoff = upper_cutoff, genotype = None, remove_circ_misaligned = remove_circ_misaligned)
    pooled_data, T_arr = group_data_by_const_rout(RTs_dict, tpt_inds = None, tpts = None)
    pooled_data = [1/p_data_t[~np.isnan(p_data_t)]*scale_factor for p_data_t in pooled_data]

    return pooled_data, T_arr    

def get_WSU_data(lower_cutoff = 167, upper_cutoff = 10000, scale_factor = 1000, include_SD = True, include_controls = False):

    module_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(get_WSU_data)))
    RTs_dict = json_to_dict(f'{module_dir}/WSU_pvt_data_RTs.json')
    metadata_dict = json_to_dict(f'{module_dir}/WSU_pvt_data_metadata.json')
    times = json_to_dict(f'{module_dir}/WSU_pvt_data_times.json')
    # import pdb; pdb.set_trace()
    if not include_controls:
        RTs_dict = {id: RTs_dict[id] for id in RTs_dict.keys() if metadata_dict[id]['Condition'] == 'Deprivation'}
        tpt_inds = np.arange(len(times['8003']))
        tpts = times['8003']
    if not include_SD:
        RTs_dict = {id: RTs_dict[id] for id in RTs_dict.keys() if metadata_dict[id]['Condition'] == 'Control'}
        tpt_inds = np.arange(len(times['8155']))
        tpts = times['8155']

    RTs_dict = {id: [np.array(session_RTs) for session_RTs in sessions] for id, sessions in RTs_dict.items()}
    RTs_dict = filter_data(RTs_dict, lower_cutoff, upper_cutoff, genotype = None, remove_adaptation_effect = False, remove_circ_misaligned = False)

    pooled_data, T_arr = group_data_by_const_rout(RTs_dict, tpt_inds = tpt_inds, tpts = tpts)
    pooled_data = [1/p_data_t[~np.isnan(p_data_t)]*scale_factor for p_data_t in pooled_data]

    return pooled_data, T_arr

def subsample_data_arr(data, num_samples, sample_size = None, rng = None, replace_subsamples = False, replace_within_subsamples = False):

    if sample_size is None:
        sample_size = int(np.floor(len(data)/num_samples))
    if (sample_size > len(data)/num_samples) and (not replace_within_subsamples):
        sample_size = int(np.floor(len(data)/num_samples))
        print("Warning: cannot draw samples of size > len(data)/num_samples without replacement.")
    if sample_size > len(data):
        sample_size = len(data)
        print("Warning: requested sample size is greater than number of data points, only using number of data points.")
    if rng is None:
        rng = np.random.default_rng(12345)
    #import pdb; pdb.set_trace()
    
    if not replace_subsamples and not replace_within_subsamples: # no replacement
        shuffled_data = rng.permutation(data)[:num_samples*sample_size]
        subsampled_data = shuffled_data.reshape(num_samples, sample_size)

    elif replace_subsamples and not replace_within_subsamples: # no replacement within a subsample, but all data is available for each subsample
        subsampled_data = np.zeros((num_samples, sample_size))
        for i in range(num_samples):
            subsampled_data[i] = rng.choice(data, size = sample_size, replace = False)

    else: # every data point is sampled with replacement
        subsampled_data = rng.choice(data, size = (num_samples, sample_size), replace = True)
    # for i in range(num_samples):

    #     inds = rng.choice(np.arange(0, len(data_t)), size = sample_size, replace = replace)
    #     subsampled_data[i] = data_t[inds]
    #     data_t = np.delete(data_t, inds)
        
    return subsampled_data

def subsample_pooled_data(data, num_samples, sample_size = None, rng = None, replace_subsamples = False, replace_within_subsamples = False):

    if rng is None:
        rng = np.random.default_rng(12345)

    subsampled_data = []
    for data_t in data:

        # sample_size = int(np.floor(len(data_t)/num_samples))
        subsampled_data.append(subsample_data_arr(data_t, num_samples, sample_size, rng, replace_subsamples, replace_within_subsamples))
        # subsampled_data_t = np.zeros((num_samples, sample_size))

        # for i in range(num_samples):

        #     inds = rng.choice(np.arange(0, len(data_t)), size = sample_size, replace = replace)
        #     subsampled_data_t[i] = data_t[inds]
        #     data_t = np.delete(data_t, inds)
        
        # subsampled_data.append(subsampled_data_t)
    
    return subsampled_data

def get_HC_data(npts, return_all = False):

    HC_data = scio.loadmat('../data/pvtCRfit.mat')['pvtCRfit']
    # H_t_all = (HC_data[0][0][1].T)[0][120:].T
    # C_t_all = (HC_data[0][0][2].T)[0][120:].T
    H_t_all = (HC_data[0][0][1].T)[0].T
    C_t_all = (HC_data[0][0][2].T)[0].T
    
    session_inds = np.rint(np.linspace(0, len(C_t_all), npts)).astype(int) # get points corresponding to each DLMO time for fitting
    if session_inds[-1] >= len(C_t_all):
        session_inds[-1] -= 1

    C_t = C_t_all[session_inds]
    H_t = H_t_all[session_inds]

    if return_all:
        return C_t_all, H_t_all
    else:
        return C_t, H_t


def get_50hr_HC_data(DLMO_ind = 1310, return_all = False):

    HC_data = scio.loadmat('../data/HC_50_hrs.mat')['HC']
    C_t_all = (HC_data[0][0][1].T)[0]
    H_t_all = (HC_data[0][0][2].T)[0]
    
    session_inds = DLMO_ind + 120*np.arange(-7, 14, 1)
    C_t = C_t_all[session_inds]
    H_t = H_t_all[session_inds]

    if return_all:
        return C_t_all, H_t_all

    return C_t, H_t

def get_ind_fit_data_arr(return_BL = True, return_SD = True, ind_keys = None):
    
    'Convenience function to load individual fits for BL and SD data.'
    BL_fits_dict = json_to_dict(f'../data/RM_repar_individuals_sessions-1-8.json')
    del BL_fits_dict["sessions"]
    SD_fits_dict = json_to_dict(f'../data/RM_repar_individuals_sessions-9-20.json')
    del SD_fits_dict["sessions"]

    if ind_keys is None:
        ind_keys = list(BL_fits_dict.keys())

    BL_pars_dict = {key: BL_fits_dict[key]["parameters"] for key in ind_keys}
    SD_pars_dict = {key: SD_fits_dict[key]["parameters"] for key in ind_keys}
    BL_pars_arr = np.array(list(BL_pars_dict.values()))
    SD_pars_arr = np.array(list(SD_pars_dict.values()))

    if return_BL and not return_SD:
        pars_arr = BL_pars_arr
    elif return_SD and not return_BL:
        pars_arr = SD_pars_arr
    elif return_BL and return_SD:
        pars_arr = np.array((BL_pars_arr, SD_pars_arr))
    else:
        raise ValueError("At least one of return_BL or return_SD must be True.")
    
    return ind_keys, pars_arr
        

def get_smooth_cdf(x, data, cutoff_freq = 1.0, filter_order = 3):

    '''Obtain a smooth'''

    #cdf = np.array([np.sum((data <= xval)) for xval in x])/len(data)
    cdf = stats.ecdf(data).cdf.evaluate(x)
    dx = x[1] - x[0]
    b, a = sig.butter(filter_order, cutoff_freq, fs = 1/dx)
    smooth_cdf = sig.filtfilt(b, a, cdf , method="pad")

    smooth_cdf = smooth_cdf/np.nanmax(smooth_cdf) # renormalise

    return smooth_cdf

def get_smooth_pdf(x, data, cutoff_freq = 1.0, filter_order = 3):
    
    smooth_cdf = get_smooth_cdf(x, data, cutoff_freq = cutoff_freq, filter_order = filter_order)
    dx = x[1] - x[0]
    d_dx = FinDiff(0, dx)
    pdf = d_dx(smooth_cdf)

    return pdf

def get_zeros(x, arr, return_inds = False):
    
    lower_zero_inds = np.nonzero(arr[:-1]*np.roll(arr, -1)[:-1] <= 0)[0]
    zero_bounding_inds = np.array([lower_zero_inds, lower_zero_inds + 1])
    f = np.abs(arr[zero_bounding_inds[0]]/(arr[zero_bounding_inds[1]] - arr[zero_bounding_inds[0]]))
    zeros = x[zero_bounding_inds[0]] + (x[zero_bounding_inds[1]] - x[zero_bounding_inds[0]])*f

    if return_inds:
        return zeros, zero_bounding_inds[0]
    else:
        return zeros

def get_derivative_zeros(x, arr, order = 1, return_inds = False):

    dx = x[1] - x[0]
    d_dx = FinDiff(0, dx, order)
    dn_arr = d_dx(arr)

    return get_zeros(x, dn_arr, return_inds)

def estimate_empirical_peaks(pdf, data, x):

    # pdf = gaussian_kde(data).evaluate(x)
    zeros, zero_inds = get_derivative_zeros(x, pdf, return_inds = True)
    pdf_vals = pdf[zero_inds]

    if len(zeros) == 1:
        return zeros, np.array([1.0]), pdf_vals
    elif len(zeros) % 2 == 0:
        import pdb; pdb.set_trace()
    elif pdf_vals[0] > pdf_vals[1]:
        peak_locs = zeros[::2]
        pdf_vals = pdf_vals[::2]
        bounds = np.concatenate((np.array([np.min(x)]), zeros[1::2], np.array([np.max(x)])))
    else:
        peak_locs = zeros[1::2]
        pdf_vals = pdf_vals[1::2]
        bounds = np.concatenate((np.array([np.min(x)]), zeros[::2], np.array([np.max(x)])))
    
    bounds = np.array([bounds[:-1], bounds[1:]]).T
    proportions = np.array([np.sum((data > lb)*(data <= ub)) for lb, ub in bounds])/len(data)

    return peak_locs, proportions, pdf_vals
