from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
import glob
from PIL import Image

class ContrastiveLearningDataset:
    def __init__(self, args, root_folder, adain):
        self.args = args
        self.root_folder = root_folder
        self.adain = adain

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.args,
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views,
                                                                  adain=self.adain),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.args,
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views,
                                                              adain=self.adain),
                                                          download=True),
                           #'imagenet': lambda: datasets.ImageNet(self.root_folder, split='unlabeled',
                           #                                 transform=ContrastiveLearningViewGenerator(
                           #                                     self.args,       
                           #                                     self.get_simclr_pipeline_transform(96),
                           #                                     n_views,
                           #                                     adain=self.adain),
                           #                                     download=True),
                            'imagenet' : lambda: ImageNet(transform=ContrastiveLearningViewGenerator(
                                                            self.args,
                                                            self.get_simclr_pipeline_transform(96),
                                                            n_views,
                                                            adain=self.adain))                                                                
                                                                }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()



class ImageNet(Dataset):
    def __init__(self, transform=None):
        self.datapath = '../../data/imagenet/n*/*.JPEG'
        self.images = glob.glob(self.datapath)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(str(self.images[idx])).convert('RGB')
        tensor = self.transform(img)
        return tensor
    
    def __len__(self):
        return len(self.images)