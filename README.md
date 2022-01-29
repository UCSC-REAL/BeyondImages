# Label Noise Transition Matrix Estimation for Tasks with Lower-Quality Features
Estimate the noise transition matrix with f-mutual information.


This code is a PyTorch implementation of the paper:


## Requirements
* Python3
* Pytorch
* Pandas
* Numpy
* Scipy
* Sklearn

## Quick Run on UCI datasets:
```
nohup bash ./run.sh > result.log &
```

## Get results:
```
cat result.log | grep "Error" 
```

## NLP benchmarks:

* Prepare BERT embeddings
* Change DataLoader to TextDataLoader
