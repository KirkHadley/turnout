# first convert to sequences and get polling data
# then run HMM
comp = [8.65,8.65,12.125,12.125,7.1125,7.1125,7.76666666666667,7.76666666666667,9.5,9.5,4.2,4.2,4.83333333333333,4.83333333333333]
comp = np.split(np.array(comp), 14) 

#rows = df.ix[df.columns[34:88]].values
#rows1 = map(np.array, map(lambda x: np.split(x, 18), rows))
#seq_data = map(lambda x: np.column_stack([x, competitivity]), rows1)
raw_seq = df[df.columns[30:72]]
raw_seq = raw_seq[sorted[raw_seq.columns]]
data = map(lambda x: np.column_stack([x, comp]),map(np.array, map(lambda x: np.split(x, 14), raw_seq.values)))




