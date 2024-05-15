import torch
import tqdm
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as transforms


def load_imagenet1k(is_train=True):

    dataset = load_dataset('imagenet-1k')
    
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
            ])
    
    def transforms_fn(examples):
        examples["pixel_values"] = [transform(image.convert('RGB')) for image in examples["image"]]
        del examples["image"]

        return examples
    
    if is_train:
        dataset = concatenate_datasets([dataset['train'], dataset['validation']])
        dataset = dataset.with_transform(transforms_fn)
        #dataset.set_format(type='torch', columns=['image','label'])
    else:
        dataset = dataset['test']
        dataset = dataset.with_transform(transforms_fn)
        #dataset.set_format(type='torch', columns=['image','label'])
    return dataset
