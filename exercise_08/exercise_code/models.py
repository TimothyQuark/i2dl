import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #
        # Hint: You can use nn.Sequential() to define your encoder.            #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             #
        # Look online for the APIs.                                            #
        # Hint: wrap them up in nn.Sequential().                               #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        ########################################################################

        self.latent_dim = self.hparams["latent_dim"]
        self.input_size = self.hparams["input_size"]

        # More layers seems to result in less overfitting, and better accuracy. 2 layers seems best
        self.encoder = nn.Sequential(

            nn.Linear(self.input_size, self.hparams["encoder_hidden"]),
            nn.BatchNorm1d(self.hparams["encoder_hidden"]),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout_p"]),

            nn.Linear(self.hparams["encoder_hidden"], self.hparams["encoder_hidden"] // 2),
            nn.BatchNorm1d(self.hparams["encoder_hidden"] // 2),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout_p"]),

            nn.Linear(self.hparams["encoder_hidden"] // 2, self.latent_dim),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        # TODO: More sophisticated decoder. Should sort of mirror the encoder,
        # hence need same input and output dimensions

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams["latent_dim"], self.hparams["decoder_hidden"]),
            nn.BatchNorm1d(self.hparams["decoder_hidden"]),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout_p"]),

            nn.Linear(self.hparams["decoder_hidden"], self.hparams["input_size"]),
            # nn.Softmax()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        reconstruction = self.encoder(x)
        reconstruction = self.decoder(reconstruction)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        # TODO: Set autoencoder and classifier to have their own optimization parameters
        self.optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams['learning_rate'],
                                 weight_decay=self.hparams["weight_decay"]
                                 )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similraly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #
        ########################################################################

        # Set model to train
        self.encoder.train()
        self.decoder.train()
        # TODO: can I just use self.train() ?

        self.optimizer.zero_grad() # Reset the gradients for every batch
        images = batch # Get images and labels from batch
        images = images.to(self.device) # Send data to device, set images and labels to device data

        images = images.view(images.shape[0], -1) # Flatten images

        pred = self.forward(images) # Stage 1: Forward
        # Compute loss. Since AE is trying to reconstruct original data (i.e. forward is encoder and decoder forward passes),
        # this is loss between original imager and prediction
        loss = loss_func(pred, images)
        loss.backward() # Stage 2: Backward
        self.optimizer.step() # Stage 3: Update parameters

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        # Set model to eval
        self.encoder.eval()
        self.decoder.eval()
        # TODO: Can i just use self.eval() ?

        loss = 0
        with torch.no_grad():
            images = batch
            images = images.to(self.device)

            images = images.view(images.shape[0], -1)
            pred = self.forward(images)
            # Compute loss. Since AE is trying to reconstruct original data (i.e. forward is encoder and decoder forward passes),
            # this is loss between original imager and prediction
            loss = loss_func(pred, images)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #
        # block of fully connected layers.                                     #
        ########################################################################


        # Noticed that my classifier often overfit training data, so using dropouts
        self.model == nn.Sequential(
            nn.Linear(encoder.latent_dim, self.hparams["classifier_hidden"]),
            nn.BatchNorm1d(self.hparams["classifier_hidden"]),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout_p"]),

            nn.Linear(self.hparams["classifier_hidden"], self.hparams["classifier_hidden"] // 2),
            nn.BatchNorm1d(self.hparams["classifier_hidden"] // 2),
            nn.ReLU(),
            nn.Dropout(self.hparams["dropout_p"]),


            nn.Linear(self.hparams["classifier_hidden"] // 2,
                      self.hparams["num_classes"]),
            # nn.Softmax()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams['learning_rate'],
                                 weight_decay=self.hparams["weight_decay"]
                                 )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):

        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
