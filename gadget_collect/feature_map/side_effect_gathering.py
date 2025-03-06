import os.path
import pickle
import glob
import numpy as np
import apg.inpatients as inpatients


def get_all_feature_names(file_path):
    with open(file_path, 'r') as f:
        feature_name_list = [x.strip() for x in f.readlines()]
    return np.array(feature_name_list)

final_dict= {}
full_features = get_all_feature_names('selected_10000_features.txt')
with open('realizable_features.txt','r') as source:
    for line in source.readlines():
        feat_index = int(line.split("\t")[0].strip())
        feature = line.split("\t")[1].strip().split("::")[1]
        print(feature)
        final_dict[feat_index] = []
        orgs, new_mask_features = inpatients.fetch_harvested([feature])
        for org in orgs:
            for k in org.feature_dict:
                # print(k)
                fea_idx_ = np.where(np.char.endswith(full_features, k) == True)[0]
                if len(fea_idx_) > 0:
                    final_dict[feat_index].append(fea_idx_[0])
        print(final_dict)

with open("output.p", 'wb') as f:
    pickle.dump(final_dict, f)