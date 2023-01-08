import logging

import torch

from torchvision import transforms, datasets
from .dataset import *
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from PIL import Image
from .autoaugment import AutoAugImageNetPolicy
import os

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    if args.dataset == 'dogs':

        if args.sm_vit:
            train_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                # transforms.RandomHorizontalFlip(), !!! FLIPPING in dataset.py !!!
                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                                        
            test_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            train_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                transforms.RandomCrop((args.img_size, args.img_size)),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                                        
            test_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                transforms.CenterCrop((args.img_size, args.img_size)),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        trainset = dogs(args.dataset, 
                        root=args.data_root,
                        is_train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False,
                        sm_vit=args.sm_vit,
                        low_memory=args.low_memory,
                        img_size=args.img_size
                        )
        testset = dogs(args.dataset, 
                        root=args.data_root,
                        is_train=False,
                        cropped=False,
                        transform=test_transform,
                        download=False,
                        sm_vit=args.sm_vit,
                        low_memory=args.low_memory,
                        img_size=args.img_size
                        )


    elif args.dataset== "CUB":

        if args.sm_vit:
            train_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size),Image.BILINEAR),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                #transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

            test_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size),Image.BILINEAR),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            train_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                transforms.RandomCrop((args.img_size, args.img_size)),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                                            
            test_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                transforms.CenterCrop((args.img_size, args.img_size)),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        

        trainset = eval(args.dataset)(args.dataset, root=args.data_root, is_train=True, \
            transform=train_transform, sm_vit=args.sm_vit, low_memory=args.low_memory, img_size=args.img_size)
        testset = eval(args.dataset)(args.dataset, root=args.data_root, is_train=False, \
            transform = test_transform, sm_vit=args.sm_vit, low_memory=args.low_memory, img_size=args.img_size)


    elif args.dataset == 'nabirds':

        if args.sm_vit:
            train_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add (from FFVT) mb try?
                #transforms.RandomHorizontalFlip(), # !!! FLIPPING in dataset.py !!!

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                        
                ])

            test_transform=transforms.Compose([
                transforms.Resize((args.img_size, args.img_size), Image.BILINEAR),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                          
                ])
        else:
            train_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                transforms.RandomCrop((args.img_size, args.img_size)),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                        
                ])

            test_transform=transforms.Compose([
                transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                transforms.CenterCrop((args.img_size, args.img_size)),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                          
                ])            


        trainset = NABirds(args.dataset, root=args.data_root, is_train=True, \
            transform=train_transform, sm_vit=args.sm_vit, low_memory=args.low_memory, img_size=args.img_size)
        testset = NABirds(args.dataset, root=args.data_root, is_train=False, \
            transform=test_transform, sm_vit=args.sm_vit, low_memory=args.low_memory, img_size=args.img_size)




    ### Not optimised datasets:

    if args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)        
    

    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )


    elif args.dataset== "air":
        train_transform=transforms.Compose([transforms.Resize((args.resize_size, args.resize_size),Image.BILINEAR),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # my add
                                    transforms.RandomHorizontalFlip(),
                                    #transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size), Image.BILINEAR),
                                    transforms.CenterCrop((args.img_size, args.img_size)),
                                    #transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize([0.8416, 0.867, 0.8233], [0.2852, 0.246, 0.3262])])
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = FGVC_aircraft(root=args.data_root, is_train=True, transform=train_transform)
        testset = FGVC_aircraft(root=args.data_root, is_train=False, transform = test_transform)
    


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
