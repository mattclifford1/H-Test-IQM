# Pipeline
Overall goal: to get a score/distribution from a given dataset and compare against another

## Data
We want to be able to work with either numpy or torch datasets. The format for each is:

 - numpy: list of numpy images
 - torch: iterable DataLoader (batch, W, H, 3)

 Current numpy datasets:
  - from h_test_IQM.datasets.numpy_loaders import kodak_loader

Current torch datasets:
  - from h_test_IQM.datasets.torch_loaders import CIFAR10_loader



## Distortions (Optional)
Transform image data e.g. unit noise. Will work with either numpy or torch data (return the same type)


## Models/scores
Input either numpy or torch. Must accept batch of torch (can convert to single numpy under the hood)

Return numpy score (batch, N) where N is the number of features of the score (usually N=1)


## Hypothesis test
Currently just plot them or get the KL
TODO the full test methodology



# Putting it all together

Get distributions blueprint:
```
for (target, test):
    - Dataloader:                          i -> image data (numpy/torch)
    - Distortion:   image data (numpy/torch) -> image data (numpy/torch)
    - Models:       image data (numpy/torch) -> score (numpy)

    return (target_dist, test_dist)
```

Test blueprint: 
```
compare (target_dist, test_dist)
```