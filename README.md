# H-Test-IQM

## Setting up Environment
First install this repo
```
git clone https://github.com/mattclifford1/H-Test-IQM
cd H-Test-IQM
```
Then make a python env e.g. with conda
```
conda create -n h_test python=3.10 -y
conda activate h_test
```

Install pytorch for GPU if required e.g.
```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#### conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```


Install as editable package
```
pip install -e .
```

## Running code/ experiments
Go to the notebooks folder