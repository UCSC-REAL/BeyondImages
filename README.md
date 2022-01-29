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
* Change [DataLoader](https://github.com/UCSC-REAL/Est-T-MI/blob/daecabcf734decd25ca52b7af1b1ed3635bc9ab1/utils/dataloader.py#L8) to [TextDataLoader](https://github.com/UCSC-REAL/Est-T-MI/blob/daecabcf734decd25ca52b7af1b1ed3635bc9ab1/utils/dataloader.py#L134)
* [Use](https://github.com/UCSC-REAL/Est-T-MI/blob/daecabcf734decd25ca52b7af1b1ed3635bc9ab1/runner.py#L5) the corresponding dataset name and run (more details will be available soon)
