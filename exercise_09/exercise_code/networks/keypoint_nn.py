"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

# Timm helper functions

# Output dimensions of maxpool layer
def dim_out_maxpool(input_dim, kernel, stride=None, padding=0, dilation=1):
    if stride == None:
        stride = kernel
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

# Output dimensions of convolution layer
def dim_out_conv(input_dim, kernel, stride=1, padding=0, dilation=1):
    return (input_dim + 2 * padding - dilation *( kernel -1 ) - 1) // stride + 1

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server

        """
        super().__init__()
        self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################


        # TODO: Weight initialization and batchnorms

        self.device = hparams.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))


        # TODO: maxpooling, dropout, init weights for linear layer, batchnorm
        # TODO: Probably a good idea to rescale input too be much smaller so we can have more layers. Use maxpool straight away?

        # Calculate out dimensions of input maxpool
        dim_out_maxin = dim_out_maxpool(
            input_dim=self.hparams['input_size'],
            kernel=self.hparams['input_pool_kernel'],
            stride=None,
            padding=self.hparams['input_pool_padding']
        )

        # Calculate out dimension of first convolution layer
        dim_out_conv1 = dim_out_conv(input_dim=dim_out_maxin,
                                     kernel=self.hparams['conv1_kernel'],
                                     stride=self.hparams['conv1_stride'],
                                     padding=self.hparams['conv1_padding'])
        dim_out_max1 = dim_out_maxpool(
            input_dim=dim_out_conv1,
            kernel=self.hparams['conv1_pooling_kernel'],
        )

        # Calculate out dimension of second convolution layer
        dim_out_conv2 = dim_out_conv(input_dim=dim_out_max1,
                                     kernel=self.hparams['conv2_kernel'],
                                     stride=self.hparams['conv2_stride'],
                                     padding=self.hparams['conv2_padding'])
        dim_out_max2 = dim_out_maxpool(
            input_dim=dim_out_conv2,
            kernel=self.hparams['conv2_pooling_kernel'],
        )

        # Calculate out dimension of third convolution layer
        dim_out_conv3 = dim_out_conv(input_dim=dim_out_max2,
                                     kernel=self.hparams['conv3_kernel'],
                                     stride=self.hparams['conv3_stride'],
                                     padding=self.hparams['conv3_padding'])
        dim_out_max3 = dim_out_maxpool(
            input_dim=dim_out_conv3,
            kernel=self.hparams['conv3_pooling_kernel'],
        )


        # Calculate out dimension of fourth convolution layer
        # dim_out_conv4 = dim_out_conv(input_dim=dim_out_max3,
        #                              kernel=self.hparams['conv4_kernel'],
        #                              stride=self.hparams['conv4_stride'],
        #                              padding=self.hparams['conv4_padding'])
        # dim_out_max4 = dim_out_maxpool(
        #     input_dim=dim_out_conv4,
        #     kernel=self.hparams['conv4_pooling_kernel'],
        # )


        # Model loosely based on https://arxiv.org/pdf/1710.00977.pdf

        self.model = nn.Sequential(

            # Because we have a limited num of parameters, we will need to maxpool the input already
            nn.MaxPool2d(
                kernel_size=self.hparams['input_pool_kernel'],
                stride=None,
                padding=self.hparams['input_pool_padding']
            ),

            # Layer 1
            nn.Conv2d(
                in_channels=1, out_channels=self.hparams['conv1_out_channels'],
                kernel_size=self.hparams['conv1_kernel'],
                stride=self.hparams['conv1_stride'],
                padding=self.hparams['conv1_padding']
            ),
            nn.BatchNorm2d(self.hparams['conv1_out_channels']),
            nn.ReLU(),
            # nn.Dropout(p=self.hparams['conv1_dropout']),
            nn.MaxPool2d(
                kernel_size=self.hparams['conv1_pooling_kernel'],
                stride=None,
                padding=0
            ),


            # Layer 2
            nn.Conv2d(
                in_channels=self.hparams['conv1_out_channels'], out_channels=self.hparams['conv2_out_channels'],
                kernel_size=self.hparams['conv2_kernel'],
                stride=self.hparams['conv2_stride'],
                padding=self.hparams['conv2_padding']
            ),
            nn.BatchNorm2d(self.hparams['conv2_out_channels']),
            nn.ReLU(),
            # nn.Dropout(p=self.hparams['conv2_dropout']),
            nn.MaxPool2d(
                kernel_size=self.hparams['conv2_pooling_kernel'],
                stride=None,
                padding=0
            ),


            # Layer 3
            nn.Conv2d(
                in_channels=self.hparams['conv2_out_channels'], out_channels=self.hparams['conv3_out_channels'],
                kernel_size=self.hparams['conv3_kernel'],
                stride=self.hparams['conv3_stride'],
                padding=self.hparams['conv3_padding']
            ),
            nn.BatchNorm2d(self.hparams['conv3_out_channels']),
            nn.ReLU(),
            # nn.Dropout(p=self.hparams['conv3_dropout']),
            nn.MaxPool2d(
                kernel_size=self.hparams['conv3_pooling_kernel'],
                stride=None,
                padding=0
            ),


            # Layer 4
            # nn.Conv2d(
            #     in_channels=self.hparams['conv3_out_channels'], out_channels=self.hparams['conv4_out_channels'],
            #     kernel_size=self.hparams['conv4_kernel'],
            #     stride=self.hparams['conv4_stride'],
            #     padding=self.hparams['conv4_padding']
            # ),
            # # nn.BatchNorm2d(self.hparams['conv4_out_channels']),
            # nn.ELU(),
            # nn.MaxPool2d(
            #     kernel_size=self.hparams['conv4_pooling_kernel'],
            #     stride=None,
            #     padding=0
            # ),
            # nn.Dropout(p=self.hparams['conv4_dropout']),

            # Linear layers
            nn.Flatten(),

            nn.Linear( dim_out_max3 * dim_out_max3 * self.hparams['conv3_out_channels'], self.hparams['linear_weights']),
            nn.BatchNorm1d(self.hparams['linear_weights']),
            nn.ReLU(),
            # nn.Dropout(p=self.hparams['linear_dropout']),

            # nn.Linear( self.hparams['linear_weights'], self.hparams['linear_weights']),
            # # nn.BatchNorm1d(self.hparams['linear_weights']),
            # nn.ELU(),
            # nn.Dropout(p=0.6),

            # Final output layer
            nn.Linear( self.hparams['linear_weights'], self.hparams['output_size']),
            nn.Tanh() # Normalize output to -1 +1 (images are normalized)
        )

        # Believe you need to set the optimizer after the network has been defined, else self.parameters()
        # is an empty generator
        self.set_optimizer()

        # Initialize the weights for the linear layers
        with torch.no_grad():
            for layer in self.model:
                if type(layer) == nn.Linear:
                    # print(model.bias)
                    torch.nn.init.kaiming_uniform_(
                        layer.weight, nonlinearity='relu')
                    # model.bias.data.fill_(0.01)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################


        return self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    # Timm code starts here
    def set_optimizer(self):

        self.optimizer = None
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hparams['learning_rate'],
                                          weight_decay=self.hparams["weight_decay"]
                                          )

    def training_step(self, batch, loss_func):

        # batch is a dict: {'image', 'keypoints'}
        # Set the model to train
        self.model.train()

        self.optimizer.zero_grad() # Reset the gradient for every batch
        images = batch['image'] # get images of the batch
        keypoints = batch['keypoints'] # get keypoints of the batch
        keypoints = keypoints.view(images.shape[0], -1)  # Flatten keypoints from XY coords to a vector
        # Send data to device, set images to device data
        images = images.to(self.device)

        pred = self.forward(images) # Send the 2D image to the model

        loss = loss_func(pred, keypoints) # Compute loss of model predictions to actual keypoints (ground truth)
        loss.backward() # Stage 2: Backward
        self.optimizer.step() # Stage 3: Update Parameters

        return loss

    def validation_step(self, batch, loss_func):

        loss = 0

        # Set model to eval
        self.model.eval()
        with torch.no_grad():
            images = batch['image']
            images.to(self.device)
            keypoints = batch['keypoints']
            keypoints = keypoints.view(images.shape[0], -1)  # Flatten keypoints from XY coords to a vector

            pred = self.forward(images)
            loss = loss_func(pred, keypoints)

        return loss

    # Timm code ends here


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
