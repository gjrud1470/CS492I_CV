# 2020 Fall CS492I CV project Team4 Repository

## Introduction
- This repository is for 2020 Fall CS492I CV project of Team 4.  
- In this [```README.md```](https://github.com/HyeongshinYoon/CS492I_CV/blob/master/README.md) we mainly explain about our **Main model with Highest Accuracy**.  
- You can check the explanation about *Best score model* or *Models that we tried but not used* in other branches.
	### Members
	    Hyunjin Kim 20170191
		Hyeongshin Yoon 20170438
		Kyoung Hur 20180717
***

## Directory Structure
<pre><code>CS492I_CV
|- ImageDataLoader.py
|- main.py
|- models.py
|- setup.py
|- README.md
|- main_model.tar.gz
</code></pre>
## Requirements
### NSML Installation
<pre><code>1) Download the compressed file from the [link](https://ai.nsml.navercorp.com/download) below
    https://ai.nsml.navercorp.com/download
2) Decompress the file
    $ tar -xvzf < downloaded archive >
3) Register installed path to PATH
    $ export PATH=$PATH:< NSML_ROOT_PATH >
</code></pre>
### Package Installation
- All packages/libraries that we used is defined in [```setup.py```](https://github.com/HyeongshinYoon/CS492I_CV/blob/master/setup.py)

## Dataset
- We use ```NAVER fashion_dataset``` for training and evaluation.
## Path to Pre-trained Model 
- It is the session and checkpoint name of **Main model with highest accuracy**
<pre><code>kaist004/fashion_dataset/829 MixSim_10w_best
</code></pre>
#### Unpack to pre-trained model from Github ( [```link```](https://github.com/HyeongshinYoon/CS492I_CV/blob/master/main_model.tar.gz) )
<pre><code>$ git clone https://github.com/HyeongshinYoon/CS492I_CV.git
$ tar -xzvf main_model.tar.gz</pre></code>

#### Download pre-trained model from NSML
<pre><code>nsml model pull kaist004/fashion_dataset/829 MixSim_10w_best ./
</code></pre>

## Commands to train/test
> ### 1. Before Training/Testing
> <pre><code> 1) Register installed path to PATH
>   $ export PATH=$PATH:< NSML_ROOT_PATH >
> 
> 2) Login to NSML
>   $ nsml login
>   -> Then put your github ID & password
> </code></pre>
> 
> ### 2. For Training
> #### Full Command for Training our Main Model
> <pre><code>$ nsml run -d fashion_dataset \
>   -a "--batchsize 500 --ema_optimizer_lr 0.005 --seed 50 --gpu_ids 0,1" \
>   -g 2
> </code></pre>
> 1) Run a NSML session
> <pre><code>$ cd < your working dir >
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
> <pre><code>$ nsml rm -f [SESSION]
> </code></pre>
> #### Optional Arguments
> - Specific informations bellow is about the arguments that we used in our experiments.   
> All arguments are defined in ```main.py```
	<pre><code>--epochs		    The number of training epochs (Default : 800)
	--batchsize		    The size of batch (Default : 140)
	--pre_train_epoch	The number of pre-training epochs (Default : 400)
	--fine_tune_epoch 	The number of fine-tuning epochs (Default : 30)
	--seed			    The random value of seed (Default : 123)
	--optimizer_lr 		The learning rate for optimizer (Default : 1e-2)
	--ema_optimizer_lr	The learning rate for EMA optimizer (Default : 1e-4)
	--gpu_ids		    The name of GPUs (Default : '0')</code></pre>
> 
> ### 3. For Testing
> <pre><code>nsml submit [Options] [SESSION_NAME] [CHECKPOINT]
>   (eg. nsml submit kaist004/fashion_dataset/1085 MixSim_best)
> 
>   * Options
>   -t, --test : Submit the session as debug mode(test mode) to find the errors from dataset or codes before actual submission
> </code></pre>

***

## Result
- About MixSim(Main model)
  - Use SimCLR pre-training & fine-tuning with contrastive loss 
  - Difference between Base Model
    - Data Augmentation : Add ColorJitter & GrayScale, Remove VerticalFlip
    - Dropout : 0.3 (Base Model : 0.5)
    - Learning Rate scheduler : CosineAnnealing (Base Model : None)
    - Optimizer : YOGI (Base Model : Adam)
  
|Model|Network|GPUs|Batch size|Epochs|pre_train_epoch|fine_tune_epoch|optimizer_lr|ema_optimizer_lr|acc_top1(%)|acc_top5(%)|Session (checkpoint)|
|:------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Base|ResNet18|1|200|200|-|-|0.0001|0.0001|10.90|21.25|kaist004/fashion_dataset/53 (Res18MM_best)
|MixSim (Main model)|ResNet18|2|500|800|400|30|0.01|0.005|27.92|50.19|kaist004/fashion_dataset/829 (MixSim_10w_best)