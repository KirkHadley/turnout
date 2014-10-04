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
pres_years = map(lambda x: np.repeat(x, 3), filter(lambda x: x % 4 == 0, range(2000,2014,2)))
sen_years = map(lambda x: np.repeat(x, 3), filter(lambda x: x % 4 != 0, range(2000,2014,2)))
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


print 'finished printing, now mapping'
raw_seq = pd.read_csv('testing.csv', usecols = a)
comp = [8.65,8.65,12.125,12.125,7.1125,7.1125,7.76666666666667,7.76666666666667,9.5,9.5,4.2,4.2,4.83333333333333,4.83333333333333]
comp = np.split(np.array(comp), 14)

#rows = df.ix[df.columns[34:88]].values
#rows1 = map(np.array, map(lambda x: np.split(x, 18), rows))
#seq_data = map(lambda x: np.column_stack([x, competitivity]), rows1)
#raw_seq = df[df.columns[30:72]]
raw_seq = raw_seq[sorted(raw_seq.columns)]
data = map(lambda x: np.column_stack([x, comp]),map(np.array, map(lambda x: np.split(x, 14), raw_seq.values)))
training = [data[:10000], data[10000:20000],data[20000:30000], data[30000:40000],data[40000:50000], data[50000:60000], data[70000:80000], data[80000:90000], data[90000:100000]]
del raw_seq
pool = Pool(12)
res = pool.map(makeGaussHMM,training)


def makeGaussHMM(d):
    new_mod = GaussianHMM(4, n_iter = 10000)
    new_results = new_mod.fit(d)
    return new_results


#l = range(10)
#pool = Pool(10)
#res = pool.map(makeGaussHMM,l)
#results = res.get()
print 'finished modeling'
    
#competitivity = np.split(np.array(competitivity), 18)
#rows = df.ix[df.columns[34:88]].values
#rows1 = map(np.array, map(lambda x: np.split(x, 18), rows))
#seq_data = map(lambda x: np.column_stack([x, competitivity]), rows1)
#data = map(lambda x: np.column_stack([x, competitivity]),map(np.array, map(lambda x: np.split(x, 18),df.ix[:4][df.columns[30:42]].values)))
 
