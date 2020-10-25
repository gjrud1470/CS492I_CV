# 2020 Fall CS492I CV project Team4 Repository

## Introduction
- This repository is for 2020 Fall CS492I CV project of Team 4.  
- In this ```README.md``` we mainly explain about our **Main model with Highest Accuracy**.  
- You can check the explanation about *Best score model* or *Models that we tried but not used* in other branches.
***

## Directory Structure
<pre><code>
CS492I_CV
|- ImageDataLoader.py
|- main.py
|- models.py
|- setup.py
</code></pre>
## Requirements
### NSML Installation
<pre><code>
1) Download the compressed file from the link below
    https://ai.nsml.navercorp.com/download
2) Decompress the file
    $ tar -xvzf < downloaded archive >
3) Register installed path to PATH
    $ export PATH=$PATH:< NSML_ROOT_PATH >
</code></pre>
### Package Installation
- All packages/libraries that we used is defined in ```setup.py```

## Dataset
- We use ```NSML fashion_dataset``` for training and evaluation.
## Path to Pre-trained Model 
- It is the path of **Main model with highest accuracy**
<pre><code>
check pull kaist004/fashion_dataset/829 MixSim_10w_best
</code></pre>

## Commands to train/test
> ### 1. Before Training/Testing
> <pre><code>
> 1) Register installed path to PATH
>   $ export PATH=$PATH:< NSML_ROOT_PATH >
> 
> 2) Login to NSML
>   $ nsml login
>   -> Then put your github ID & password
> </code></pre>
> 
> ### 2. For Training
> #### Full Command for Training our Main Model
> <pre><code>
> $ nsml run -d fashion_dataset \
>   -a "--batchsize 500 --ema_optimizer_lr 0.005 --seed 50 --gpu_ids 0,1" \
>   -g 2
> </code></pre>
> 1) Run a NSML session
> <pre> <code>
> $ cd < your working dir >
> $ nsml run -d [dataset]
>   (eg. $ nsml run -d fashion_dataset)
> 
>   * Running Options
>   -g : Define the number of GPUs 
>       (eg. $ nsml run -d fashion_dataset -g 2)
>   -a : Define the value of arguments
>       (eg. $ nsml run -d fashion_dataset \ 
>              -a "--pre_train_epoch 200 --fine_tune_epoch 30 --epochs 1000 --batchsize 500")
>   -m : Write the message for session
>       (eg. $ nsml run -d fashion_dataset -m "Mix-Sim model with ResNet50")
> </code></pre>
> 2) Terminate NSML session
> <pre><code>
> $ nsml rm -f [SESSION]
> </code></pre>
> #### Optional Arguments
> - Specific informations bellow is about the arguments that we used in our experiments.   
> All arguments are defined in ```main.py```
> > <pre><code>
> > --epochs : Define the number of training epochs
> > --batchsize : Define the size of batch
> > --pre_train_epoch : Define the number of pre-training epochs
> > --fine_tune_epoch : Define the number of fine-tuning epochs
> > --seed : Define the random value of seed
> > --optimizer_lr : Define the learning rate for optimizer
> > --ema_optimizer_lr : Define the learning rate for EMA optimizer
> > --gpu_ids : Define the name of GPUs
> > </code></pre>
> 
> ### 3. For Testing
> <pre><code>
> $ nsml submit [Options] [SESSION_NAME] [CHECKPOINT]
>   (eg. nsml submit kaist004/fashion_dataset/1085 MixSim_best)
> 
>   * Options
>   -t, --test : Submit the session as debug mode(test mode) to find the errors from dataset or codes before actual submission
> </code></pre>

***

## Result
|Model|Network|batch size|epochs|pre_train_epoch|fine_tune_epoch|optimizer_lr|ema_optimizer_lr|acc_top1(%)|acc_top5(%)|Session (checkpoint)|
|------|---|---|---|---|---|---|---|---|---|---|
|Base|ResNet18|200|200|-|-|0.0001|0.0001|10.90|21.25|kaist004/fashion_dataset/53 (Res18MM_best)
|MixSim (Main model)|ResNet18|500|800|400|30|0.01|0.005|27.92|50.19|kaist004/fashion_dataset/829 (MixSim_10w_best)