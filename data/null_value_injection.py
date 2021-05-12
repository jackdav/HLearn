import numpy as np
import pandas as pd
import random

X = np.loadtxt("income-x.csv", delimiter=',', dtype='str', encoding='utf-8-sig')

'''
income is 5k tuples

so reserve 1k for clean testing
'''

big_count = 0
count = 0
for i in range(X.shape[0]):
    if i < 4000:
        for j in range(X[i].shape[0]):
            big_count += 1
            # 5% chance of null value
            rand = random.randint(0,19)
            if rand == 0:
                # first column has to be an index column
                if j != 0:
                    # third column can't be null
                    if j != 2:
                        # sixth column can't be null
                        if j != 5:
                            # update tuple if matches all criteria
                            count += 1
                            X[i][j] = "NULL"
# 40000
print(big_count)
# ~ 1900
print(count)
pd.DataFrame(X).to_csv("income_null-x.csv", header=None, index=False)