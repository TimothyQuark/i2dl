"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        # Hint: Google the following algorithm:                                #
        # "convert-a-number-range-to-another-range-maintaining-ratio"          #
        ########################################################################


        # This function should return a rescaled copy, not modify the original memory
        old_range = self._data_max - self._data_min
        new_range = self.max - self.min

        new_images = np.copy(images)

        # When we iterate over a list directly in Python, we essentially work on
        # a copy (no concept of pointers, references or ownership fml) So
        # iterate over range of images length, and use that to modify the images
        for i in range(len(new_images)):
            new_images[i] = (((new_images[i] - self._data_min) * new_range) / old_range) + self.min

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return new_images


def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape (C,).
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint 1: You can use numpy to calculate the mean and standard         #
    # deviation.                                                           #
    #                                                                      #
    # Hint 2: Make sure that the shapes of the resulting arrays are (C,)   #
    # and not [1, C], [C, 1] or anything else. Use print(mean.shape) to    #
    # test yourself.                                                       #
    ########################################################################

    # They are using float64 here and not just float wtffffff
    # print(type(images[0]))

    # Original method, concatenate channels into new tensors and then take mean of entire
    # tensor
    # channel_0 = np.zeros((images[0].shape[0], images[0].shape[1], len(images)), dtype=type(images[0][0, 0, 0]))
    # channel_1 = np.zeros((images[0].shape[0], images[0].shape[1], len(images)), dtype=type(images[0][0, 0, 0]))
    # channel_2 = np.zeros((images[0].shape[0], images[0].shape[1], len(images)), dtype=type(images[0][0, 0, 0]))

    # for idx, image in enumerate(images):
    #     channel_0[:, :, idx] = image[:, :, 0]
    #     channel_1[:, :, idx] = image[:, :, 1]
    #     channel_2[:, :, idx] = image[:, :, 2]

    # This code is slower than my original
    # mean = [np.mean(channel_0), np.mean(channel_1), np.mean(channel_2)]
    # std = [np.std(channel_0), np.std(channel_1), np.std(channel_2)]

    # mean = np.mean(images, axis=(0,1,2))
    # std = np.std(images, axis=(0,1,2))

    # This method is fastest by far, almost half the time needed compared to above
    mean = [np.mean(images[:, :, :, 0]), np.mean(images[:, :, :, 1]), np.mean(images[:, :, :, 2])]
    std = [np.std(images[:, :, :, 0]), np.std(images[:, :, :, 1]), np.std(images[:, :, :, 2])]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################

        images = (images - self.mean) / self.std

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
