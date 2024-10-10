import sys
sys.path.append('../sota_paper_implementation/convertors')
from damnets2tgt import convert_damnets_sample_to_tgt
samples = convert_damnets_sample_to_tgt("../sota_paper_implementation/damnets-tagGen-dymonds/experiment_files/DAMNETS_test_encovid_2024-Aug-18-17-47-45_41144/sampled_ts.pkl")
print(samples)