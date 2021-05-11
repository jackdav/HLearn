import numpy as np
import pandas as pd
import random

X = np.loadtxt("income-x.csv", delimiter=',', dtype='str', encoding='utf-8-sig')

big_count = 0
count = 0
for i in range(X.shape[0]):
    if i < 4000:
        for j in range(X[i].shape[0]):
            big_count += 1
            rand = random.randint(0,19)
            if rand == 0:
                if j != 0:
                    if j != 2:
                        if j != 5:
                            count += 1
                            X[i][j] = "NULL"

print(big_count)
print(count)
pd.DataFrame(X).to_csv("income_null-x.csv", header=None, index=False)