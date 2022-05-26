import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from . import net
from .function import adaptive_instance_normalization, coral

np.random.seed(0)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(args, vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content = content.to(args.device)
    style = style.to(args.device)

    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(args.device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, args, base_transform, n_views=2, adain=False):
        self.args = args
        self.base_transform = base_transform
        self.n_views = n_views
        self.adain = adain
        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.vgg.eval()
            self.decoder.eval()
            self.vgg.load_state_dict(torch.load('./data_aug/pretrained_model/vgg_normalised.pth'))
            self.decoder.load_state_dict(torch.load('./data_aug/pretrained_model/decoder.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda')
            self.decoder.to('cuda')

            self.content_tf = test_transform(256, True) #content_size(512), crop
            self.style_tf = test_transform(256, True) #content_size(512), crop

            self.all_style_images = glob.glob('../../data/painterbyNumbers/train/*')
            self.style_images=[]
            for si in self.all_style_images:
                if os.path.getsize(si)<1024*1024: #only select less than 1MB
                    self.style_images.append(si)
            print('all style images #{} => selected style images #{} <1MB'.format(len(self.all_style_images), len(self.style_images)))
            
            self.toPIL = transforms.ToPILImage()
            

    def __call__(self, x):
        if self.adain:
            outputs=[]
            for i in range(self.n_views):
                content = self.content_tf(x)
                style_path = self.style_images[random.randint(0,len(self.style_images)-1)] #randomly select style image
                style = self.style_tf(Image.open(str(style_path)).convert('RGB'))
                style = style.unsqueeze(0)
                content = content.unsqueeze(0)

                alpha = random.uniform(0,1)
                with torch.no_grad():
                    output = style_transfer(self.args, self.vgg, self.decoder, content, style, alpha)
                output = output.squeeze(0) #b,c,h,w-->c,h,w tensor [0-1]
                output = self.toPIL(output) #c,h,w [0-1] tensor --> h,w,c [0-255] PILImage
                outputs.append(self.base_transform(output))
            return outputs
        else:       
            return [self.base_transform(x) for i in range(self.n_views)]
