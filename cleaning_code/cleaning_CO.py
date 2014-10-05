import numpy as np
import operator
import pandas as pd
from sklearn.hmm import GaussianHMM
from multiprocessing import Pool
desired_columns = ['prefix', 'suffix', 'gender', 'born_at', 'demo', 'party', 'is_active_voter', 'is_perm_absentee', 'registered_at', 'is_do_not_call', 'residential_address2', 'residential_city', 'county', 'residential_zip5', 'residential_zip4', 'mailing_state', 'vh00g1', 'vh00p1', 'vh02g1', 'vh02p1', 'vh04g1', 'vh04p1', 'vh06g1', 'vh06p1', 'vh08g1', 'vh08p1', 'vh10g1', 'vh10p1', 'vh12g1', 'vh12p1']
df = pd.io.parsers.read_csv("../Data/CO.csv", usecols = desired_columns)
print 'read data'
df['registered_at'] = pd.to_datetime(df['registered_at'])
df[df.columns[16:30]] = df[df.columns[16:30]].fillna(-1)
df[df.columns[16:30] + 'binary_plus_registration'] = df[df.columns[16:30]]
df[df.columns[16:30] + 'binary_plus_registration'] = df[df.columns[16:30]].replace([0, 1, 2, 3, 4, 5, 6, 7,8,9], [-1, 1, 1,1,1,1,1,1,1,1])
print 'converted to binary'
df[df.columns[16:30] + '__method'] = df[df.columns[16:30]].replace([0,1, 5, 6, 8,9], [-1,0, 2, 3, 2,3])
print 'added methods'
df[df.columns[16:30] + 'party'] = df[df.columns[16:30]].replace([-1,0,2,3,4,5,6,7,8,9], [0,0,0,0,1,1,1,-1,-1,-1])
print 'added party'
pres_years = map(lambda x: np.repeat(x, 2), filter(lambda x: x % 4 == 0, range(2000,2014,2)))
sen_years = map(lambda x: np.repeat(x, 2), filter(lambda x: x % 4 != 0, range(2000,2014,2)))
years = sorted(reduce(operator.add, map(lambda x: x.tolist(), reduce(operator.add, (pres_years, sen_years)))))
election_dates = map(lambda x: pd.to_datetime(x), map(lambda x: '11/2/' + x, map(lambda x: str(x), years)))
time_since_registration = map(lambda x: x - df.registered_at, election_dates)
for i in range(14):
    df[df.columns[i + 16] + '_time_registered'] = time_since_registration[i]

# for not yet registered in voting roll
for i in df.columns[16:30]:
    df.ix[df[i + '_time_registered'] < 0, i + 'binary_plus_registration'] = 0


print 'calculated time since registration'

#TODO: get polling data
print 'writing to file'
df.to_csv('missing_polls.csv')
