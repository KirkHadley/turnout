import numpy as np
import pandas as pd
from pybrain.datasets import SequentialDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.tools.neuralnets import NetworkWriter
from sklearn.hmm import GaussianHMM
from multiprocessing import Pool
import pickle
from math import floor


# read file
# convert it
# train rnn on it
# combine rnn weights




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

comp = [8.65,8.65,12.125,12.125,7.1125,7.1125,7.76666666666667,7.76666666666667,9.5,9.5,4.2,4.2,4.83333333333333,4.83333333333333]
comp = np.split(np.array(comp), 14)




def cleanpar(x):
    df = pd.read_csv('df' + str(x), usecols = a)
    df = df[sorted(df.columns)]
    data = [ ]
    for x in df.values:
        data.append((np.column_stack([np.split(x, 14), comp)))
    ds = SequentialDataSet(3, 1)
    for i in a:
        ds.newSequence()
        for y in i:
            ds.addSample([y[0],y[2],y[3]], y[0])
    rnn = buildNetwork( ds.indim, 5, ds.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)
    trainer = RPropMinusTrainer( rnn, dataset=ds, verbose=True )
    for i in xrange(20):
        trainer.trainEpochs(2)
        training = 100. * (1.0-testOnSequenceData(rnn, ds))
        print "train error: %5.2f%%" % training
    NetworkWriter 
    return trainer, rnn, training 

pool = Pool(10)
    
res = pool.map(cleanpar, range(10))
pool.close()
pool.join()
outfile = open('RNN.pkl', 'wb')
pickle.dump(res, outfile)

         
