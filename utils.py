from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.datasets import CIFAR10, CIFAR100

from random import sample
import cv2
import numpy as np


class CIFARSelf(CIFAR10):
    def __init__(self, root='../data', train=True, transform=None, classes = None): #子类属性
        super().__init__(root=root, train=train, transform= transform)    #父类属性
        selected_indices = [id for id in range(len(self.targets)) if self.targets[id] in classes]
        self.data = self.data[selected_indices]
        self.targets = [self.targets[id] for id in selected_indices]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CIFARSup(CIFAR10):
    # dataloader where pairs of positive samples are randomly sampled from pairs
    # of inputs with the same label.
    def __init__(self, root='../data', train=True, transform=None, classes=None):
        super().__init__(root=root, train=train,transform= transform)

        selected_indices = [id for id in range(len(self.targets)) if self.targets[id] in classes]
        self.data = self.data[selected_indices]
        self.targets = [self.targets[id] for id in selected_indices]


    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]

        index_example_same_label = sample(self.get_labels(target), 1)[0]
        img2 = self.data[index_example_same_label]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        pos_1 = self.transform(img1)
        pos_2 = self.transform(img2)

        return pos_1, pos_2, target

    def get_labels(self,i):
        return [index for index in range(len(self.targets)) if self.targets[index] == i]





class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


def get_dataset(dataset_name, classes, root='../data'):
    if dataset_name == 'self':
        train_data = CIFARSelf(root=root, train=True, transform=train_transform, classes=classes)
        memory_data = CIFARSelf(root=root, train=True, transform=test_transform, classes=classes)
        test_data = CIFARSelf(root=root, train=False, transform=test_transform, classes=classes)

    elif dataset_name == 'sup':
        train_data = CIFARSup(root=root, train=True, transform=train_transform, classes=classes)
        memory_data = CIFARSup(root=root, train=True, transform=test_transform, classes=classes)
        test_data = CIFARSup(root=root, train=False, transform=test_transform, classes=classes)
    else:
        raise Exception('Invalid dataset name')

    return train_data, memory_data, test_data
