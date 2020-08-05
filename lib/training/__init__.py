from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# from PIL import Image
import torch
import random
from torch import is_tensor

CROP_TYPE = ["center", "random", "sliding"]


class Crop(object):
    """Crops the given img tensor.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_frac: crop fraction to crop from the image
    """

    def __init__(self, crop_type=None, crop_frac=1.0, sliding_crop_position=None):
        assert crop_frac <= 1.0, "crop_frac can't be greater than 1.0"
        if sliding_crop_position is not None:
            # max positions are fixed to 9
            assert sliding_crop_position < 9

        assert (
            crop_type is None or crop_type in CROP_TYPE
        ), "{} is not a valid crop_type".format(crop_type)

        self.crop_type = crop_type
        self.crop_frac = crop_frac
        self.sliding_crop_position = sliding_crop_position

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image to be cropped.
        Returns:
        """
        assert img is not None, "img should not be None"
        assert is_tensor(img), "Tensor expected"
        h = img.size(1)
        w = img.size(2)
        h2 = int(h * self.crop_frac)
        w2 = int(w * self.crop_frac)
        h_range = h - h2
        w_range = w - w2

        if self.crop_type == "sliding":
            assert self.sliding_crop_position is not None
            row = int(self.sliding_crop_position / 3)
            col = self.sliding_crop_position % 3
            x = col * int(w_range / 2)
            y = row * int(h_range / 2)

        elif self.crop_type == "random":
            x, y = random.randint(0, w_range), random.randint(0, h_range)

        elif self.crop_type == "center":
            y = int(h_range / 2)
            x = int(w_range / 2)

        if self.crop_type is not None:
            img = img.narrow(1, y, h2).narrow(2, x, w2).clone()

        return img

    def update_sliding_position(self, sliding_crop_position):
        assert (
            sliding_crop_position >= 0 and sliding_crop_position < 9
        ), "Only 9 sliding positions supported"
        self.sliding_crop_position = sliding_crop_position


# forwards input through model to get probabilities
def get_probs(model, imgs, output_prob=False):
    softmax = torch.nn.Softmax(1)
    # probs = torch.zeros(imgs.size(0), n_classes)
    imgsvar = torch.autograd.Variable(imgs.squeeze(), volatile=True)
    output = model(imgsvar)
    if output_prob:
        probs = output.data.cpu()
    else:
        probs = softmax.forward(output).data.cpu()

    return probs


# calls get_probs to get predictions
def get_labels(model, input, output_prob=False):
    probs = get_probs(model, input, output_prob)
    _, label = probs.max(1)
    return label.squeeze()
