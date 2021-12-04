# Transfer How Much: A Fine-Grained Measure of the Knowledge Transferability of User Behavior Sequences in Social Network


# Environment

Tensorflow 1.4.0


Python 3.6.3





## Background
The TEMCS is a metric to measure the level of knowledge transferability between the source and target behaviors, which is model-agnostic and acquires no training compared to other transferability metrics. Then we can apply it to select the source behavior, and to improve the performance for transfer learning-based and multi-behavior fusion-based methods. 



## Files Introduction 
1. DA_code                                              -- The implementation of the basic method of transferring learning we choose in the paper
2. Multi_bheavior_code_update                           -- The implementation of the basic method of multi-behavior fusion methods we choose in the paper
3. centers_20.csv / ... / centers_500.csv               -- The clustering results of Zhihu dataset in the paper
4. train_test_set_5.pkl / train_test_set_510.pkl        -- The train and test set of Zhihu dataset in the paper
5. TEMCS_computer.py                                    -- The TEMCS score obtained by the clustering results and the training set.
6. analysis_weight.py                                   -- The overall TEMCS score for a train set.



## Usage

1. Calculating the TEMCS score from the source behavior to the target behavior characteristics in each dataset

```sh
>>> python TEMCS_computer.py
```

2. Calculating the distribution of  the TEMCS score of a data set
```sh
>>> python analysis_weight.py
```
