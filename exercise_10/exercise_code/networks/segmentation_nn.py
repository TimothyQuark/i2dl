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
    dropout_flag=False,
    maxpool_flag=True
):
    layers = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)])
    if norm_flag:
        layers.append(nn.BatchNorm2d(out_channels))
    if active_flag:
        layers.append(nn.PReLU())
    if dropout_flag:
        layers.append(nn.Dropout2d(p=dropout_p))
    if maxpool_flag:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


def convT_b(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dropout_p=0.1,
    upsample=2,
    active_flag=True,
    norm_flag=True,
    dropout_flag=False,
    avgpool_flag=False,
    upsample_flag=True
):
    layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups)])
    if norm_flag:
        layers.append(nn.BatchNorm2d(out_channels))
    if active_flag:
        layers.append(nn.PReLU())
    if dropout_flag:
        layers.append(nn.Dropout2d(p=dropout_p))
    if avgpool_flag:
        layers.append(nn.AvgPool2d(2))
    if upsample_flag:
        layers.append(nn.Upsample(scale_factor=upsample))

    return nn.Sequential(*layers)


def weights_init(m):
    # TODO: weight init for linear layers
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
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
        
        # Submission server may nbot have v3 according to TA
        # mobilenet = models.mobilenet_v3_small(weights='DEFAULT', num_classes=23)
        
        # We on Pytorch 0.12 so only 1 pretrained to choose from
        self.encoder = models.mobilenet_v2(pretrained=True).features

        # Use for debugging
        # self.encoder.append(Print_layer())
        # summary(self, input_size=(hparams["batch_size"], 3, 240, 240))

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
            convT_b(in_channels=1280, out_channels=32 * 6, kernel_size=3, stride=2, padding=1, upsample=2, norm_flag=True),
            convT_b(in_channels=32 * 6, out_channels=32 * 5, kernel_size=3, stride=2, padding=1, upsample=2, norm_flag=True),
            convT_b(in_channels=32 * 5, out_channels=32 * 3, kernel_size=3, stride=1, padding=1, upsample=2, norm_flag=True),
            convT_b(in_channels=32 * 3, out_channels=23, kernel_size=1, stride=1, padding=0, upsample=2, norm_flag=True),
            
            nn.Upsample(size = 240),
            
            # Softmax learns super slowly, not sure if it results in something better
            # nn.Softmax(dim=1), # Calculate softmax along dim of classifier

        )

        self.device = hp.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()
        
        # Apply for my decoder but not for pretrained encoder!
        self.decoder.apply(weights_init)

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

        # See shape of input
        # print(x.shape)
        
        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # Timm code

    def set_optimizer(self):

        self.optimizer = None
        
        # Note that training encoder increases model size, so be careful!
        # Disable this to train encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hp['learning_rate'],
                                          weight_decay=self.hp["weight_decay"]
                                          )


    def training_step(self, batch, loss_func):

        # We train the decoder but not the encoder (may change later). But train() does not freeze weights,
        # of encoder, this is done using requires_grad in set_optimizer
        self.encoder.train()
        self.decoder.train()

        self.optimizer.zero_grad() # Reset gradient every batch

        images = batch[0].to(self.device) # batch x channels x H x W
        labels = batch[1].to(self.device) # Each pixel assigned a label (ground truth), batch x H x W

        # Pred is batch x num_classifiers x H x W, each pixel assigned a value that it is a
        # classifier. When evaluating, checks which classifier channel is biggest, and compares to
        # ground truth (i.e. the labels tensor)
        pred = self.forward(images)

        # Compute loss between model predictions and labels (ground truth). For CE loss, pass tensor of batch x num_classes x H x W
        # it will calculate the loss using the label tensor automatically (no need to do hot encoding or stuff like that myself)
        loss = loss_func(pred, labels) # Need to return float and not long because CrossEntropyLoss doesn't implement it for long
        loss.backward()
        self.optimizer.step()

        return loss

    def validation_step(self, batch, loss_func):
        
        loss = 0
        
        # Set model to eval
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device) # Each pixel assigned a label (ground truth)
            
            pred = self.forward(images)
            loss = loss_func(pred, labels)
            
        return loss


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
