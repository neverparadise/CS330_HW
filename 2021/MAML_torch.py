import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import misc
import imageio
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)
# logger.setLevel(logging.CRITICAL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ! Loss utilities

writer = SummaryWriter()

def cross_entropy_loss(pred, label, k_shot):
    loss = torch.nn.CrossEntropyLoss(pred, label)
    loss /= k_shot
    return loss


def accuracy(labels, predictions):
    equals = torch.eq(labels, predictions).long()
    equals = equals.long()
    accuracy = equals.sum() / equals.numel()
    return accuracy


class ConvLayers(nn.Module):
    def __init__(self, channels, dim_hidden, dim_output, img_size, kernel_size, stride):
        """ img_size = (W, H) ? """
        super(ConvLayers, self).__init__()
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.stride = stride

        W = self.img_size[0]
        H = self.img_size[1]
        K = self.kernel_size
        S = self.stride

        self.conv_layers = nn.ModuleList(
            nn.Conv2d(channels, dim_hidden, K, S, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_hidden, dim_hidden, K, S, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_hidden, dim_hidden, K, S, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_hidden, dim_hidden, K, S, bias=True),
            nn.ReLU(),
            nn.Flatten()
        )

        def conv2d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
            print((size - (kernel_size - 1) - 1) // stride + 1)
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(W, K, S)
        convw = conv2d_size_out(convw, K, S)
        convw = conv2d_size_out(convw, K, S)
        convw = conv2d_size_out(convw, K, S)

        convh = conv2d_size_out(H, K, S)
        convh = conv2d_size_out(convh, K, S)
        convh = conv2d_size_out(convh, K, S)
        convh = conv2d_size_out(convh, K, S)

        self.fc_layers = nn.ModuleList(
            nn.Linear(dim_hidden * convw * convh, dim_output),
            nn.softmax()
        )

    def forward(self, x):
        hidden = self.conv_layers(x)
        output = self.fc_layers(hidden)
        return output


def get_images(paths, labels, n_samples=None, shuffle=True):
  """
  Takes a set of character folders and labels and returns paths to image files
  paired with labels.
  Args:
    paths: A list of character folders
    labels: List or numpy array of same length as paths
    n_samples: Number of images to retrieve per character
  Returns:
    List of (label, image_path) tuples
  """
  if n_samples is not None:
    def sampler(x): return random.sample(x, n_samples)
  else:
    def sampler(x): return x
  images_labels = [(i, os.path.join(path, image))
                   for i, path in zip(labels, paths)
                   for image in sampler(os.listdir(path))]
  if shuffle:
    random.shuffle(images_labels)
  return images_labels


def image_file_to_array(filename, dim_input):
  """
  Takes an image path and returns numpy array
  Args:
    filename: Image filename
    dim_input: Flattened shape of image
  Returns:
    1 channel image
  """
  image = imageio.imread(filename)
  image = image.reshape([dim_input])
  image = image.astype(np.float32) / 255.0
  image = 1.0 - image
  return image



""" 
omniglot_resized folder??? ???????????? ????????? ????????? ??????.

???????????? ??? ????????? 20?????? ???????????? ??????. 
omniglot_resized
+---language1
+---+---character1
+---+---+---01.png
+---+---+---02.png
...
+---+---+---20.png
...
...

+---+---characterN
+---+---+---01.png
+---+---+---02.png
...
+---+---+---20.png

+---language2
...
...

????????? ??????????????? ????????? N way K shot ????????????. 
?????? meta-train??? meta-test?????? 5-way 2shot ????????? ?????????????????? ??????????????? ????????? ???????????????????
?????? ?????? ??????????????? meta-train??? meta-test??? ???????????? ??? ?????????.
????????? meta-train??? ????????????, ???????????????, meta-test??? ????????????, ?????????????????? ???????????? ??????. 
 
"""
#TODO: task distribution??? ?????? custom dataset, custom dataloader ?????????
#TODO 1: ?????? ??????????????? meta-train??? meta-test??? ????????? 
#TODO 2: meta-train?????? support set??? query set?????? ?????????
#TODO 3: meta-test?????? support set??? query set?????? ?????????


class OmniglotDataset(Dataset):
    def __init__(self, num_classes, num_samples_per_class,
                 num_meta_test_classes, num_meta_test_samples_per_class, config={}):
        """
        Args:
        num_classes: Number of classes for classification (K-way)
        num_samples_per_class: num samples to generate per class in one batch
        num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
        num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
        batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        
        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        """
        family ~ language
        """
        character_folders = []
        for family in os.listdir(data_folder):
            if os.path.isdir(os.path.join(data_folder, family)):
               for character in os.listdir(os.path.join(data_folder, family)):
                   if os.path.isdir(os.path.join(data_folder, family, character)):
                       character_folders.append(os.path.join(data_folder, family, character))

        print(character_folders)
        random.seed(123)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
    # ?????? ????????? : ?????? ?????? ???, ?????? underbar 2?????? ?????? ?????????
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # ?????????????????? ?????? ???????????? ???????????? ?????? (x, y)??? ???????????? ?????????
        return self.x_data[idx], self.y_data[idx]


class OmniglotDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(OmniglotDataLoader, self).__init__(*args, **kwargs)
    
    def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
        batch_type: meta_train/meta_val/meta_test
        shuffle: randomly shuffle classes or not
        swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
        A a tuple of (1) Image batch and (2) Label batch where
        image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
        where B is batch size, K is number of samples per class, N is number of classes
        """
        
data_folder = os.curdir+'/omniglot_resized'
character_folders = []

for family in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder, family)):
        for character in os.listdir(os.path.join(data_folder, family)):
            if os.path.isdir(os.path.join(data_folder, family, character)):
                character_folders.append(os.path.join(data_folder, family, character))

logger.debug(character_folders[0:20])
random.seed(123)
random.shuffle(character_folders)
logger.debug(character_folders[0:20])