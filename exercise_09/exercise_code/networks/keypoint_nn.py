"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

# Timm helper functions

# For debugging


class Print_layer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


def conv_b(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=0,
    dropout_p=0.1,
    active_flag=True,
    norm_flag=True,
    dropout_flag=True,
    maxpool_flag=True
):
    layers = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)])
    if norm_flag:
        layers.append(nn.BatchNorm2d(out_channels))
    if active_flag:
        layers.append(nn.ReLU())
    if dropout_flag:
        layers.append(nn.Dropout2d(p=dropout_p))
    if maxpool_flag:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


def linear_b(
    in_features,
    out_features,
    dropout_p=0.1,
    active_flag=True,
    norm_flag=True,
    dropout_flag=True,
):
    layers = nn.ModuleList(
        [nn.Linear(in_features=in_features, out_features=out_features)])
    if norm_flag:
        layers.append(nn.BatchNorm1d(out_features))
    if active_flag:
        layers.append(nn.ReLU())
    if dropout_flag:
        layers.append(nn.Dropout(p=dropout_p))

    return nn.Sequential(*layers)


def weights_init(m):
    # TODO: weight init for linear layers
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

# Output dimensions of maxpool layer


def dim_out_maxpool(input_dim, kernel=2, stride=None, padding=0, dilation=1):
    if stride == None:
        stride = kernel
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

# Output dimensions of convolution layer


def dim_out_conv(input_dim, kernel, stride=1, padding=0, dilation=1):
    return (input_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


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

        self.device = hparams.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))

        # Model loosely based on https://arxiv.org/pdf/1710.00977.pdf

        self.model = nn.Sequential(

            # Using a stride of 1 to allow overlapping of kernels, and then maxpool to reduce dimensionality
            # Don't think we need padding here?

            conv_b(1, 64, 5, 1, 0, dropout_p=0.1, dropout_flag=False),
            conv_b(64, 128, 3, 1, 0, dropout_p=0.1, dropout_flag=False),
            conv_b(128, 256, 2, 1, 0, dropout_p=0.1, dropout_flag=False),
            conv_b(256, 512, 2, 1, 0, dropout_p=0.1, dropout_flag=False),
            # Print_layer(),

            nn.Flatten(),
            # Figure out dimensions by printing out last conv layer
            linear_b(4 * 4 * 512, 128, dropout_p=0.2, dropout_flag=False),
            linear_b(128, 128, dropout_p=0.2, dropout_flag=False),
            linear_b(128, 30, active_flag=False,
                     dropout_flag=False, norm_flag=False),
            nn.Tanh() # Normalize output to -1 +1 (images are normalized)

        )
        # print (self.model)

        # All linear layers init with kaiming, but last layer is tanh, so fix it
        with torch.no_grad():
            torch.nn.init.xavier_normal_(self.model[-2][0].weight)

        # Believe you need to set the optimizer after the network has been defined, else self.parameters()
        # is an empty generator
        self.set_optimizer()

        self.apply(weights_init)

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

        self.optimizer.zero_grad()  # Reset the gradient for every batch
        images = batch['image']  # get images of the batch
        keypoints = batch['keypoints']  # get keypoints of the batch
        # Flatten keypoints from XY coords to a vector
        keypoints = keypoints.view(images.shape[0], -1)
        # Send data to device, set images to device data
        images = images.to(self.device)

        pred = self.forward(images)  # Send the 2D image to the model

        # Compute loss of model predictions to actual keypoints (ground truth)
        loss = loss_func(pred, keypoints)
        loss.backward()  # Stage 2: Backward
        self.optimizer.step()  # Stage 3: Update Parameters

        return loss

    def validation_step(self, batch, loss_func):

        loss = 0

        # Set model to eval
        self.model.eval()
        with torch.no_grad():
            images = batch['image']
            images.to(self.device)
            keypoints = batch['keypoints']
            # Flatten keypoints from XY coords to a vector
            keypoints = keypoints.view(images.shape[0], -1)

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
