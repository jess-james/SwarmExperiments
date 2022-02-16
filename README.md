# SwarmExperiments
*Experiments using a 'Swarm' learner on synthetic and climate data.*

Swarm learner as based on algorithm defined in 'Online Multitask Learning with Long-Term Memory' https://arxiv.org/abs/2008.07055.

Swarm performance is compared to the Fixed- and Variable Share learners (Bosquet, Warmuth https://www.jmlr.org/papers/volume3/bousquet02b/bousquet02b.pdf)

Thesis write-up under OnlineLearning.pdf



Experiments conducted on synthetic data (in synthetic_experiments.py) and on Global Circulation models (in ta_experiments.py). 

GCMs sourced from the ‘Climate of the 20th Century Experiment’ (20C3M) for IPCC and can be accessed at https://esgf-node.llnl.gov/search/cmip3/. Learners trained on Climate Data from 1980-2000, and formatted for experiments in import_data.py. 

