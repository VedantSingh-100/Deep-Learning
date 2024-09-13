There were 2 architectures involved in HW2P2: - 
1. Convnext
2. Inception Resnet

During the various ablations, Convnext performed to a certain extent for the classification tasks but did not perform well for the verification tasks. However, Inception Resnet got the results for both- classification tasks and verification tasks. 

The details of the architectures are as follows: - 

1. Convnext: - 
Architecure design: - The architecture was designed by employing block like structures. Each block consisted of: - 

- Depthwise Convolution
- Layer normalization
- pointwise convolution
- Activation Layer(GeLU)
- A Gamma Layer
- A residual layer with drop path

I used an Inverted Bottleneck architecture with a convolution layer used before deploying any block. The expansion factor is 4. The stem layer consisted of a depthwise convolution of 4x4. I ran the model for 100 epochs with AdamW optimiser with Cosine Annealing schedulder. 

The data augmentation techniques I used were: - 
1. Random Horizontal Flip
2. Random Perspective
3. Gaussian Blur
4. Color Jitter
5. Random Erasing
6. Normalizing(mean=[0.5103, 0.4014, 0.3509], std=[0.2708, 0.2363, 0.2226])
7. Random Erasing

The number of blocks per channel division is as follows: - 
Number of Channels | Number of Blocks
       80          |        2        
       161         |        2         
       322         |        6  
       644         |        2

This was my Convnext architecture.

2. Inception Resnet
Architecture Design: - This design was inspired by the github repository: - https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inception_resnet_v2.py 

The architecture was designed by employing different blocks with different functionalities. The various blocks consisted of: - 
-> Conv2d
-> Reduction A
-> Stem
-> Inception Resnet A
-> Inception Resnet B
-> Reduction B
-> Inception resnet C

Finally all these blocks are employed in the main class Inception ResnetV2. 

The optimizer used in this was SGD with momentum = 0.9 and weight decay = 1e-4
The scheduler used in this was Consine Annealing.

The entire model ran for 50 epochs with an inital learning rate of 0.1

The data augmentation techniques used were: - 
1. Random Rotation
2. Color Jitter
3. Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



