import numpy as np

def build_set(m, dd, tup):
    null_attr = []
    cs = []
    for idx in range(len(tup)):
        if tup[idx] == 'NULL':
            null_attr.append(idx)
    if len(null_attr):
        for i in range(m):
            temp = np.copy(tup)
            for j in null_attr:
                val = np.random.choice(dd[str(j)],1)
                temp[j] = val[0]
            cs.append(temp)
    return cs

class CandidateSet:
    def __init__(self, initial_tuple):
        self.original = initial_tuple
        self.set = []
    def bulk_add_candidate(self, candidates=None):
        for x in candidates:
            self.set.append(x)
    def grab_candidate(self, idx=0):
        if idx < len(self.set):
            return self.set[idx]
        else:
            return self.set[0]

class CandidateDictionary:
    def __init__(self, m=5):
        self.set = {}
        self.m = m
        self.domain_dict = {}
    def generate_candidates(self, dataset):
        self.generate_domain_dict(dataset)
        for i in dataset:
            cs = CandidateSet(i)
            cs.bulk_add_candidate(build_set(self.m, self.domain_dict, i))
            self.set[str(i[0])] = cs
    def generate_domain_dict(self, dataset):
        tpose_ds = np.transpose(dataset)
        for idx in range(np.size(tpose_ds, 0)):
            temp_arr = np.unique(tpose_ds[idx])
            temp_arr = np.delete(temp_arr, np.where(temp_arr == 'NULL'))
            self.domain_dict[str(idx)] = temp_arr
    def grab_candidate_set(self, idx):
        return self.set[str(idx)]
    def print_dict(self):
        print(self.set)