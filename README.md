# **Offical Code for CFBNL submitted to KDD 25**

## Description
**Contrastive Functional Brain Network Learning (CFBNL)** is designed to automatically learn brain network structures according to 4 distinct research scales, namely sample-scale, subject-scale, group-scale and group-scale, respectively. The description and typical scenario of these four scales are as follow:
- Sample-Scale focuses on short time segments, essential for studying dynamic neural processes (e.g. dynamic brain network configuration).
- Subject-Scale aggregates samples from the same subject (individual/participand, etc.), preserving personal variability (e.g., brain fingerprinting).
- Group-Scale identifies shared patterns between groups (categories/classes), capturing representative group-level patterns (e.g., gender, mental disease).
- Project-Scale uncovers dataset-wide patterns, providing insights that cannot be derived from individual groups alone (e.g., brain atlas, human connectome project).
![Research Scales](./Scene.pdf)

**CFBNL** learns brain networks using method derived from Graph Structure Learning (GSL) research. Related works of GSL can be found in [Awesome GSL]{https://github.com/GSL-Benchmark/Awesome-Graph-Structure-Learning}. According to the target scale, **CFBNL** learns brain networks as follow:
- Sample-scale: **CFBNL** learns multiple brain networks for each subject according to each sample in the dataset. Each brain network depicts a transient co-activivy of variables in the sample. The brain function of a subject could be modeled by multiple brain networks (dynamic brain network).
- Subject-scale: **CFBNL** learns brain network for each subject based on their corresponding samples in the dataset. The brain network is learned by merging information from multiple homologic samples. The brain function is modeled by a single brain network.
- Grouph-scale: **CFBNL** learns a shared brain network for all the subjects based on their corresponding samples in a group. The learned brain network reflects the common property of all the subjects in the corresponding group.
- Project-scale: **CFBNL** learns a shared brain network for all the subjects based on all of their samples in a dataset. A dataset is typically collected for a project. And researchers might be curious about the common features of all the subjects in the project.

The framework of **CFBNL** is depicted as follow:
![CFBNL](./Frame.pdf)

## Brain Network Visualization on HCP Gender
[BNVis](./GenderNets.pdf)

## Environment
- pytorch
- pytorch geometric
- numpy
- sklearn

## Dataset
- [HCP]{https://www.humanconnectome.org}, [NeuroGraph]{https://anwar-said.github.io/anwarsaid/neurograph.html}
- [Cog State]{https://openneuro.org/datasets/ds004148}
- [SLIM]{https://fcon_1000.projects.nitrc.org/indi/retro/southwestuni_qiu_index.html}

## Tools for Dataset Preprocess
- [DPABI]{rfmri.org/DPABI}
- [EEGLab]{sccn.ucsd.edu/eeglab/index.php}
- 

## Use
The model can be trained and tested by the following command
```bash
python main.py
```

The hyper parameters are set in config.py
