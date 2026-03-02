from tqdm import tqdm

import numpy as np

import data_processing_funcs as dp

rng = np.random.default_rng(seed=12345) 
seed = rng.integers(2**32 - 1)
rng = np.random.default_rng(seed)

RTs_dict = dp.load_RT_data_dict()
RTs_dict = dp.filter_data(RTs_dict, lower_cutoff = 167, upper_cutoff = 10000, genotype = None, remove_adaptation_effect = True, remove_circ_misaligned = True)
n_samples = 1000

for s in tqdm(range(n_samples)):
    bootstrap_RTs_dict = {'bs_index': s}
    for i, (pid, sessions) in enumerate(RTs_dict.items()):
        bootstrap_sessions = []
        for session in sessions:
            if len(session) == 0:
                bootstrap_sessions.append(session.tolist())
                continue
            bootstrap_session = rng.choice(session, size=len(session), replace=True, shuffle=False)
            bootstrap_sessions.append(bootstrap_session.tolist())
        bootstrap_RTs_dict[pid] = bootstrap_sessions
    dp.dict_to_json(bootstrap_RTs_dict, f'../data/bootstrap_samples/RTs/bootstrap_RTs_dict_{s}.json')
