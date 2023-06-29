"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


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


class Encoder(nn.Module):

    def __init__(self, hparams, num_classes=23):
        super().__init__()

        self.hparams = hparams

        # TODO: Try out different pretrained models
        mobile_model = models.mobilenet_v3_small(weights='DEFAULT', num_classes=23)
        # print(mobile_model.features)

        self.encoder = mobile_model.features

        # Use for debugging
        # self.encoder.append(Print_layer())

        # TODO: Initialize weights

    def forward(self, x):

        return self.encoder(x)


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.encoder = Encoder(hparams=self.hp) # Feature detector, pretrained
        self.decoder = nn.Sequential(
            nn.MaxPool2d(2),
            # Print_layer(),
            nn.Flatten(),
            # Print_layer(),
            nn.Linear(4 * 4 * 576, 256),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(256, num_classes),
        )

        self.device = hp.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # Timm code

    def set_optimizer(self):

        self.optimizer = None

        # TODO: Set autoencoder and classifier to have their own optimization parameters
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hp['learning_rate'],
                                          weight_decay=self.hp["weight_decay"]
                                          )


    def training_step(self, batch, loss_func):

        # We train the decoder but not the encoder
        self.decoder.train()

        self.optimizer.zero_grad() # Reset gradient every batch

        images = batch[0].to(self.device)

        labels = batch[1].to(self.device) # Each pixel assigned a label using one hot notation

        pred = self.forward(images)

        # Compute loss between model predictions and labels (ground truth)
        losses = loss_func(pred.unsqueeze(0), labels.unsqueeze(0))





    # Timm code end

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(
            target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
