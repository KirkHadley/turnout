import numpy as np
import operator
import pandas as pd
from sklearn.hmm import GaussianHMM
from multiprocessing import Pool
import pickle
from math import floor
from sklearn.preprocessing import normalize

a = ['vh00p1binary_plus_registration',
 'vh00g1binary_plus_registration',
 'vh02p1binary_plus_registration',
 'vh02g1binary_plus_registration',
 'vh04p1binary_plus_registration',
 'vh04g1binary_plus_registration',
 'vh06p1binary_plus_registration',
 'vh06g1binary_plus_registration',
 'vh08p1binary_plus_registration',
 'vh08g1binary_plus_registration',
 'vh10p1binary_plus_registration',
 'vh10g1binary_plus_registration',
 'vh12p1binary_plus_registration',
 'vh12g1binary_plus_registration',
 'vh00p1__method',
 'vh00g1__method',
 'vh02p1__method',
 'vh02g1__method',
 'vh04p1__method',
 'vh04g1__method',
 'vh06p1__method',
 'vh06g1__method',
 'vh08p1__method',
 'vh08g1__method',
 'vh10p1__method',
 'vh10g1__method',
 'vh12p1__method',
 'vh12g1__method',
 'vh00p1party',
 'vh00g1party',
 'vh02p1party',
 'vh02g1party',
 'vh04p1party',
 'vh04g1party',
 'vh06p1party',
 'vh06g1party',
 'vh08p1party',
 'vh08g1party',
 'vh10p1party',
 'vh10g1party',
 'vh12p1party',
 'vh12g1party']

raw_seq = pd.read_csv('../cleaning_code/missing_polls.csv', usecols = a)
comp = [8.65,8.65,12.125,12.125,7.1125,7.1125,7.76666666666667,7.76666666666667,9.5,9.5,4.2,4.2,4.83333333333333,4.83333333333333]
comp = np.split(np.array(comp), 14)

raw_seq = raw_seq[sorted(raw_seq.columns)]
b = map(lambda x: np.split(x, 14), raw_seq.values)
data = map(lambda x: np.column_stack([x, comp]), b)
del b 
del raw_seq

def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l)-1, n)]

scores = map(lambda x: (res[0].score(x)+ res[1].score(x)+ res[2].score(x)+ res[3].score(x)+ res[4].score(x)+res[5].score(x)+ res[6].score(x)+ res[7].score(x)+ res[8].score(x)+ res[9].score(x))/10, map(normalize, data))

df = pd.read_csv('../cleaning_code/missing_polls.csv')
df['likelihood'] = scores
df.sort('likelihood', inplace=True)

try:
    df_list = chunks(df, 300000)   
    for i in range(len(df_list)):
        df_list[i].to_csv('chunked/df' + i)
except:
    print 'chunk failed'

print 'writing full to csv'

df.to_csv('allclean.csv')

