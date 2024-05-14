import torch
import tqdm
from datasets import load_dataset, concatenate_datasets


def load_imagenet1k(is_train):

    dataset = load_dataset('imagenet-1k')
    if is_train:
        dataset = concatenate_datasets([dataset['train'], dataset['validation']])
        dataset.set_format(type='torch', columns=['image','label'])
    else:
        dataset = dataset['test']
        dataset.set_format(type='torch', columns=['image','label'])
    return dataset