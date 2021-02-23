import numpy as np
import webcolors
from skimage.color import rgb2hsv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import os
import os.path as osp
import glob
import cv2
import random

test_num = 100
num = 400 # train number
k = 400
if k == 5:
    batchsize = 1
else:
    batchsize = 32
num = 400

groups = {}
groups[0] = list(range(10))

PATH = "/home/yucheng/Desktop/project/correcting-agent/results_0128"
model_name = "model_k{}_0".format(k)
latent_n = 4

def namedtuple_to_rgb(rgb):
    return np.array(rgb, dtype=np.float32)/255


def get_colour(colour_name):
    colour_tuple = webcolors.name_to_rgb(colour_name)
    return np.array(colour_tuple, dtype=np.float32)/255


def name_to_rgb(name):
    rgb = webcolors.name_to_rgb(name)
    return np.array(rgb, dtype=np.float32) / 255


def name_to_hsv(name):
    rgb = name_to_rgb(name)
    hsv = rgb2hsv([[rgb]])[0][0]
    return hsv


def generate_colour_generator(mean=0, std=0):
    """ Creates a colour generator function which generates random HSV colours based on the provided mean and std

    the SV channels will always use the same mean and std, ensuring that colours are not too dark or too light

    :param mean:
    :param std:
    :return: colour_generator function
    """
    def colour_generator():
        return np.array((((np.random.randn() * std + mean) % 360) / 3.6 , 100 - np.abs(np.random.randn() * 10), 100 - np.abs(np.random.rand() * 20)))/100
    return colour_generator

# These mean and std values seem to generate good values for each colour which all look sensibly like the specified colour
colour_values = {"red": (0, 5),
                 "orange": (30, 5),
                 "yellow": (58, 2),
                 "green": (120, 9),
                 "blue": (220, 13),
                 "purple": (270, 9),
                 "pink": (315, 9)}
# This dict maps a colour to its colour generator
# So to generate red use colour_generators['red']()
colour_generators = {colour: generate_colour_generator(*values) for colour, values in colour_values.items()}

class VAE(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, latent_n=1, groups={}, device="cpu"):
        super(VAE, self).__init__()
        layers = []
        self.latent_n = latent_n
        self.groups = groups
        self.groups_n = len(groups.keys())
        self.device = device
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # IMAGE ENCODER
        self.conv_channels = [32,32,64,64]
        self.dense_channels = [1024, 32]
        self.deconv_channels = [64, 64]

        kernel_size=7
        self.encoder_conv_0 = nn.Conv2d(3, self.conv_channels[0], kernel_size, padding=3, stride=2) # (32, 32)
        kernel_size=5
        self.encoder_conv_1 = nn.Conv2d(self.conv_channels[0], self.conv_channels[1], kernel_size, padding=2, stride=2) # (16, 16)
        kernel_size=3
        self.encoder_conv_2 = nn.Conv2d(self.conv_channels[1], self.conv_channels[2], kernel_size, padding=1, stride=2) # (8, 8)
        self.encoder_conv_3 = nn.Conv2d(self.conv_channels[2], self.conv_channels[3], kernel_size, padding=1, stride=2) # (4, 4)
        
        self.encoder_dense_0 = nn.Linear(self.dense_channels[0], self.dense_channels[1])
        self.encoder_mu = nn.Linear(self.dense_channels[1], self.latent_n)
        self.encoder_ln_var = nn.Linear(self.dense_channels[1], self.latent_n)

        # IMAGE DECONV DECODER
        self.decoder_dense_0 = nn.Linear(self.latent_n, self.dense_channels[1])
        self.decoder_dense_1 = nn.Linear(self.dense_channels[1], self.dense_channels[0])
        self.decoder_conv_3 = nn.Conv2d(self.conv_channels[3], self.conv_channels[2], kernel_size, padding=1)
        self.decoder_conv_2 = nn.Conv2d(self.conv_channels[2], self.conv_channels[1], kernel_size, padding=1)
        self.decoder_conv_1 = nn.Conv2d(self.conv_channels[1], self.conv_channels[1], kernel_size, padding=1)
        self.decoder_output_img = nn.Conv2d(self.conv_channels[1], 3, kernel_size, padding=1)

        # CLASSIFIERS
        self.classifiers = nn.ModuleList([nn.Linear(1, len(items)) for key, items in self.groups.items()])
        
        self.encoder = [self.encoder_conv_0,
                        self.encoder_conv_1,
                        self.encoder_conv_2,
                        self.encoder_conv_3,
                        self.encoder_dense_0,
                        self.encoder_mu,
                        self.encoder_ln_var]
        
        self.decoder = [self.decoder_dense_0,
                        self.decoder_dense_1,
                        self.decoder_conv_3,
                        self.decoder_conv_2,
                        self.decoder_conv_1,
                        self.decoder_output_img]
        
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.encoder)):
            self.encoder[i].weight.data.normal_(0, 0.01)
            
        for i in range(len(self.decoder)):
            self.decoder[i].weight.data.normal_(0, 0.01)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        
        conv_0_encoded = F.leaky_relu(self.encoder_conv_0(x))
        conv_1_encoded = F.leaky_relu(self.encoder_conv_1(conv_0_encoded))
        conv_2_encoded = F.leaky_relu(self.encoder_conv_2(conv_1_encoded))
        conv_3_encoded = F.leaky_relu(self.encoder_conv_3(conv_2_encoded))

        reshaped_encoded = torch.flatten(conv_3_encoded, start_dim=1)
        dense_0_encoded = F.leaky_relu(self.encoder_dense_0(reshaped_encoded))
        mu = self.encoder_mu(dense_0_encoded)
        logvar = self.encoder_ln_var(dense_0_encoded)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z):
        
        dense_0_decoded = self.decoder_dense_0(z)
        dense_1_decoded = self.decoder_dense_1(dense_0_decoded)
        reshaped_decoded = dense_1_decoded.view((len(dense_1_decoded), self.conv_channels[-1], 4, 4))
        up_4_decoded = torch.nn.Upsample(scale_factor=2)(reshaped_decoded)
        deconv_3_decoded = F.relu(self.decoder_conv_3(up_4_decoded))
        up_3_decoded = torch.nn.Upsample(scale_factor=2)(deconv_3_decoded)
        deconv_2_decoded = F.relu(self.decoder_conv_2(up_3_decoded))
        up_2_decoded = torch.nn.Upsample(scale_factor=2)(deconv_2_decoded)
        deconv_1_decoded = F.relu(self.decoder_conv_1(up_2_decoded))
        up_1_decoded = torch.nn.Upsample(scale_factor=2)(deconv_1_decoded)
        out_img = self.decoder_output_img(up_1_decoded)
        
        return torch.sigmoid(out_img)
    
    def predict_labels(self, z, softmax=False):
        result = []
        
        for i in range(self.groups_n):
            prediction = self.classifiers[i](z[:, i, None])

            # need the check because the softmax_cross_entropy has a softmax in it
            if softmax:
                result.append(F.softmax(prediction))
            else:
                result.append(prediction)

        return result
    
    def get_latent(self, x):
        mu, logvar, _ = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        return z
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        # img_out = self.sp_decode(z)
        img_out = self.decode(z)
        labels_out = self.predict_labels(z)
        
        return img_out, labels_out, mu, logvar 
    
    def get_loss(self):
        
        def loss(img_in, img_out, labels_in, labels_out, mu, logvar):
            
            rec = nn.MSELoss(reduction="none")(img_out, img_in)
            rec = torch.mean(torch.sum(rec.view(rec.shape[0], -1), dim=-1))

            label = 0
            for i in range(self.groups_n):
                label += nn.CrossEntropyLoss(ignore_index=100)(labels_out[i], labels_in)
                
            kld = (((-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / img_in.shape[0])

            rec *= self.alpha
            kld *= self.beta
            label *= self.gamma

            return rec + label + kld, rec, label, kld
    
        return loss


def fruit_loader(num=400):
    train_path = '/home/yucheng/Desktop/project/weak_label_lfd/fruit/Train/*'
    test_path = '/home/yucheng/Desktop/project/weak_label_lfd/fruit/test/*'
    crop_size = 64
    
    training_fruit_img = []
    training_label = []
    test_fruit_img = []
    test_label = []
    
    # load training images(#num per class)
    for dir_path in glob.glob(train_path):
        img_label = dir_path.split("/")[-1]
        count = 0
        for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image, (crop_size, crop_size))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            training_fruit_img.append(image)
            training_label.append(img_label)
            count += 1
            if count == num:
                break

    # load all test images
    for dir_path in glob.glob(test_path):
        img_label = dir_path.split("/")[-1]
        count = 0
        for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image, (crop_size, crop_size))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            test_fruit_img.append(image)
            test_label.append(img_label)
            count += 1
            if count == test_num:
                break 
            
    train_imgs = np.array(training_fruit_img)
    train_labels = np.array(training_label)
    test_imgs = np.array(test_fruit_img)
    test_labels = np.array(test_label)
    
    return train_imgs, train_labels, test_imgs, test_labels


train_imgs_100, train_labels_100, test_imgs_100, test_labels_100 = fruit_loader()

imgs = np.swapaxes(test_imgs_100, 1, 3) # (n, 3, 64, 64)
imgs = imgs/255
labels = test_labels_100

label_to_id = {v.lower():k for k,v in enumerate(np.unique(test_labels_100)) }
id_to_label = {v:k for k,v in label_to_id.items() }

net = VAE(latent_n=latent_n, groups=groups)
net.load_state_dict(torch.load(osp.join(PATH, model_name)))

def fruit_select(fc):
    select_10 = []
    fruit_dic = []

    for (fruit, fnum) in fc.items():
        while not fnum == 0:
            select_10.append(label_to_id[fruit.lower()])
            fnum -= 1
    
    incid = [np.random.randint(0,100) + 100*i for i in select_10]

    filtered_data_imgs = np.take(imgs, incid, axis=0).astype(np.float32)
    latent_out, _, _ = net.encode(torch.tensor(filtered_data_imgs))

    for i in range(10):
        fruit_name = id_to_label[select_10[i]]
        pred = latent_out[i].detach().numpy()
        colour = random.sample(colour_values.keys(), 1)
        long_data = np.hstack((pred, colour_generators[colour[0]]()))
        fruit_dic.append((fruit_name, np.array(long_data, dtype='float')))
    
    # latent = {}
    # for group_idx in groups.keys():
    #     latent[group_idx] = {}
    #     for label in range(len(groups[group_idx])):  
    #         indecies = [i for i, label_i in enumerate(select_10) if label_i == label]
    #         if  not indecies == []:
    #             filtered_data_imgs = np.take(imgs, indecies, axis=0).astype(np.float32)

    #             latent_out, _, _ = net.encode(torch.tensor(filtered_data_imgs))
    #             latent[group_idx][label] = latent_out.detach().numpy()
    
    # for fruit_id, pred_list in latent[0].items():
    #     for pred in pred_list:
    #         fruit_name = id_to_label[fruit_id]
    #         fruit_dic.append((fruit_name, pred))

    return fruit_dic