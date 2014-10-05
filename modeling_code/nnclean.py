import pandas as pd
from multiprocessing import Pool
import numpy as np
from math import floor
keep08 = ['gender',
 'age',
 'demo',
 'is_active_voter',
 'is_perm_absentee',
 'residential_zip5',
 'mailing_state',
 'is_do_not_call',
 'vh08g1_time_registered',
 'vote_ratio',
'party_strength',
'most_common_method',
'vh08g1binary_plus_registration'
]

keep10 = [
'gender',
 'age',
 'demo',
 'is_active_voter',
 'is_perm_absentee',
 'residential_zip5',
 'mailing_state',
 'is_do_not_call',
 'vh10g1_time_registered',
 'vote_ratio',
'party_strength',
'most_common_method',
'vh10g1binary_plus_registration'
]

keep12 = [
'gender',
 'age',
 'demo',
 'is_active_voter',
 'is_perm_absentee',
 'residential_zip5',
 'mailing_state',
 'is_do_not_call',
 'vh12g1_time_registered',
 'vote_ratio',
'party_strength',
'most_common_method',
'vh12g1binary_plus_registration'
]

def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l)-1, n)]

def cleanpar(x):
    df = pd.read_csv('df' + str(x))
    del df[df.columns[0]]
    df['age'] = pd.to_datetime(datetime.datetime.now()) - pd.to_datetime(df['born_at'])
    del df['born_at']
    df['gender'].replace(['F', 'M', 'U', 'nan'], [-1, 1, 0, 0], inplace = True)
    df['mailing_state'] = df['mailing_state'].fillna(0)
    l = df[df['mailing_state'] != 'CO']['mailing_state'].unique()
    df['mailing_state'] = df['mailing_state'].replace(l[1:], [-1]*len(l[1:]))
    df['mailing_state'] = df['mailing_state'].replace('CO', 1)
    df['demo'] = df['demo'].fillna(0)
    l = df[df['demo'] > 0]['demo'].unique()
    df[df['demo'] > 0] = df[df['demo'] > 0].replace(l, range(1, 1+len(l)))    
    df['is_active_voter'] = df['is_active_voter'].fillna(0)
    df['is_active_voter'] = df['is_active_voter'].replace([True, False], [1, -1])
    df['is_perm_absentee'] = df['is_perm_absentee'].fillna(0)
    df['is_perm_absentee'] = df['is_perm_absentee'].replace([True, False], [1, -1])  
    df['is_do_not_call'] = df['is_do_not_call'].fillna(0)
    df['is_do_not_call'] = df['is_do_not_call'].replace([True, False], [1, -1])  
    if x==0:
        df['vote_ratio'] = df[df.columns[10:20]].sum(axis = 1)/10
        df['party_strength'] = df[df.columns[30:35]].sum(axis =1)/5
        df['most_common_method'] = df[df.columns[20:30]].mode(axis=1)
        df.vh08g1_time_registered = df.vh08g1_time_registered.str.slice(stop = -14)
    if x==1:
        df['vote_ratio'] = df[df.columns[7:22]].sum(axis = 1)/12
        df['party_strength'] = df[df.columns[34:40]].sum(axis =1)/6
        df['most_common_method'] = df[df.columns[22:34]].mode(axis=1) 
        df.vh10g1_time_registered = df.vh10g1_time_registered.str.slice(stop = -14)
    if x==2:
        df['vote_ratio'] = df[df.columns[10:24]].sum(axis = 1)/14
        df['party_strength'] = df[df.columns[38:45]].sum(axis =1)/7
        df['most_common_method'] = df[df.columns[24:38]].mode(axis=1)
        df.vh12g1_time_registered = df.vh12g1_time_registered.str.slice(stop = -14)
    return df


print 'cleaning' 
pool = Pool(3)
res = pool.map(cleanpar,range(3))
pool.close()
pool.join()

print 'finished cleaning'
print 'now writing'
try:
    df_list = map(lambda x: chunk(x, 100000), res)
    for i in range(len(df_list)):
        for y in range(len(df_list[i])):
            df_list[i][y].to_csv('forNN/finaltraining/df' + str(i) + str(y))
    print 'finished writing'
except:
    print 'chunk failed'


print 'writing big'
for i in range(len(res)):
    res[i].to_csv('forNN/finaltraining/big/df' + str(i)



