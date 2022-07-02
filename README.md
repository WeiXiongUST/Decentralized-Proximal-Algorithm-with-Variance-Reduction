This repository contains the code used for the paper "PMGT-VR: A decentralized proximal-gradient algorithmic framework with variance reduction". The implementation is inspired by the work of Boyue Li (see https://github.com/liboyue/Network-Distributed-Algorithm) but is simplified. 

## Requirement 
- matplotlib
- networkx
- numpy

## Experiment Details

### Dataset Generalization
We generate datasets with get_data/generate_data first and conduct experiments with fixed datasets. The parameters may need to be slightly tuned due to the randomness of data generation but the main point remains the same. 

largescale means that we conduct experiments in terms of dataset with $n=64000$ instead of $n=6400$ (default).

### Gossip Matix

We conduct algorithms on three gossip matrices whose spectral gaps are (the corresponding problems are from easy to hard)
- Mild: $2.5 \times 10^{-2}$;
- Middle: $5 \times 10^{-3}$;
- Large: $5 \times 10^{-4}$.

The details of matrix generalization can be found in the function ring_graph2 in gen_matrix.py.

### Condition Number

We use two regularization parameters to control the condition number $\kappa$, referred as e5 (small $\kappa$) and e7 (large $\kappa$).


### Hyper-Parameters
The hyper-parameters are searched via grid-searching method and we tune all the involved algorithms to their best performance. We list the hyper-parameters in every exp_xx files. You may need to slightly tune them due to the randomness of dataset generalization.

### Experiment Goals

The exp_impact_nmix files explore the relation between convergence rate and the consensus steps $K$. 

Other files correspond to comparison between PMGT-VR methods and sota proximal algorithms under different settings.

## Contact
If you have any questions, feel free to drop me an email via wxiongae@connect.ust.hk
