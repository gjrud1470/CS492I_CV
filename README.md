# Tried Methods
## FixMatch
Paper : https://arxiv.org/abs/2001.07685
- FixMatch is a consistency regularization based semi-supervised learning method that performs better than MixMatch in paper.
- It uses RandAugment and two cross-entropy calculation for loss function.
- ```randaugment.py``` and ```aug_fix.py``` is the codes about the RandAugment in Fixmatch.
- Because of lack of the time, loss function is not applied yet, but if we have more time, we will apply loss function from Fixmatch.

## WideResNet
Paper : https://arxiv.org/abs/1605.07146
- WideResNet is the network focus on increasing width of network.
- It shows better performance than original ResNet.
- We applied WideResNet28-2, but it didn't show better enough performance and spends too much time, so we decided not to use WideResNet on our project.
- ```wideresnet.py``` is the code that define WideResNet.