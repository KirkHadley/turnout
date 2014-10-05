turnout
=======
Predicting individual-level voting behavior with neural nets and HMM clustering

1) train HMM on sample (parallelized)
<br>
2) score all obs on HMM, rank by log likelihood (not parallel yet, unfortunately)
3) split obs into ~10 clusters based on log likelihood ranking
4) Train separate RNN's for each cluster, using sequential data exclusively (parallelized)
5) Train 3 global feed forward neural networks for the 08, 10, and 2012 elections (parallelized)
6) Combine weights (average) for the 3 nets; higher weights for 2012 and 2010
7) Pass respective outputs from RNN and FFNN to a single-hidden-layered FFNN to ensemble intelligently
*) To test: 
          Pass test data to HMM, calculate log likelihood, rank and cluster
          Pass sequential data to appropriate RNN (RNN that belongs to the individual's cluster), get out predictions
          Pass all individuals data to FFNN, get out predictions
          Pass predictions from RNN and FFNN to ensembler
