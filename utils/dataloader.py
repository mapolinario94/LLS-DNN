import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
try:
    import pyvww
except:
    pass
__all__ = ["cifar100_dataloader", "cifar100_augmented_dataloader", "tinyimagenet_dataloader",
           "cifar10_augmented_dataloader", "cifar10_basic_dataloader",
           "imagenette_dataloader", "imagenet_dataloader", "vww_dataloader", "fashionmnist_dataloader",
           "mnist_dataloader", "imagenette_basic_dataloader"]

def vww_dataloader(batch_size=64, test_batch_size=128, path_for_dataset="./data/VWW/"):
    transform_train = transforms.Compose([
        transforms.Resize(132),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(132),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=path_for_dataset + "all2014",
                                                                annFile=path_for_dataset + "annotations/instances_train.json",
                                                                transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    test_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=path_for_dataset + "all2014",
                                                               annFile=path_for_dataset + "VWW/annotations/instances_val.json",
                                                                transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return (train_loader, test_loader, test_loader)


def imagenet_dataloader(batch_size=64, test_batch_size=128):
    labels = 1000
    traindir = os.path.join('/local/a/imagenet/imagenet2012/', 'train')
    valdir = os.path.join('/local/a/imagenet/imagenet2012/', 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2)
        ]))
    testset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(128),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=16)

    return (train_loader, test_loader, test_loader)

def imagenette_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.Resize(160),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.2)
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.Imagenette('~/Datasets', split='train', download=False, size='160px', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    traintestset = torchvision.datasets.Imagenette('~/Datasets', split='train', download=False, size='160px', transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.Imagenette('~/Datasets', split='val', download=False, size='160px', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return (train_loader, traintest_loader, test_loader)


def imagenette_basic_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.Resize(132),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.Resize(132),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.Imagenette('~/Datasets', split='train', download=False, size='160px', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    traintestset = torchvision.datasets.Imagenette('~/Datasets', split='train', download=False, size='160px', transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.Imagenette('~/Datasets', split='val', download=False, size='160px', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return (train_loader, traintest_loader, test_loader)


def tinyimagenet_dataloader(batch_size, test_batch_size=128, path_for_dataset="./data/tiny-imagenet-200/"):
    labels = 200

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    traindir = os.path.join(path_for_dataset, 'train')
    valdir = os.path.join(path_for_dataset, 'val/images')
    testdir = os.path.join(path_for_dataset, 'test/images')
    trainset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2)
        ]))
    valset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    traintest_loader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return (train_loader, traintest_loader, test_loader)


def cifar100_augmented_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.2)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100('~/Datasets', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    traintestset = torchvision.datasets.CIFAR100('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR100('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return (train_loader, traintest_loader, test_loader)


def cifar100_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100('~/Datasets', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    traintestset = torchvision.datasets.CIFAR100('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR100('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return (train_loader, traintest_loader, test_loader)


def cifar10_augmented_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.2)
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10('~/Datasets', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    traintestset = torchvision.datasets.CIFAR10('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return (train_loader, traintest_loader, test_loader)


def cifar10_basic_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10('~/Datasets', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    traintestset = torchvision.datasets.CIFAR10('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return (train_loader, traintest_loader, test_loader)



def mnist_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    trainset = torchvision.datasets.MNIST('~/Datasets', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    traintestset = torchvision.datasets.MNIST('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False)

    testset = torchvision.datasets.MNIST('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return (train_loader, traintest_loader, test_loader)


def fashionmnist_dataloader(batch_size, test_batch_size=128):

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,)),
    ])

    trainset = torchvision.datasets.FashionMNIST('~/Datasets', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    traintestset = torchvision.datasets.FashionMNIST('~/Datasets', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=test_batch_size, shuffle=False)

    testset = torchvision.datasets.FashionMNIST('~/Datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    return (train_loader, traintest_loader, test_loader)




