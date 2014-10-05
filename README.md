turnout
=======
Predicting individual-level voting behavior with neural nets and HMM clustering

1. train HMM on sample (parallelized)

2. score all obs on HMM, rank by log likelihood (not parallel yet, unfortunately)
<br>
3. split obs into ~10 clusters based on log likelihood ranking
<br>
4. Train separate RNN's for each cluster, using sequential data exclusively (parallelized)
<br>
5. Train 3 global feed forward neural networks for the 08, 10, and 2012 elections (parallelized)
<br>
6. Combine weights (average) for the 3 nets; higher weights for 2012 and 2010
<br>
7. Pass respective outputs from RNN and FFNN to a single-hidden-layered FFNN to ensemble intelligently
<br>
8. To test:
    *Pass test data to HMM, calculate log likelihood, rank and cluster
    
          *Pass sequential data to appropriate RNN (RNN that belongs to the individual's cluster), get out predictions
          *Pass all individuals data to FFNN, get out predictions
          *Pass predictions from RNN and FFNN to ensembler
