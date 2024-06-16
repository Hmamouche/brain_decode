import json
import numpy
import re

from transformers import logging
logging.set_verbosity_error()

from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

import cv2

from CLIP import clip

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset):
        super(ImageCaptioningDataset, self).__init__()
        self.dataset = dataset
        self.prompt = ""
        self.max_words = 200

        #self.transform = transforms.Compose([ transforms.ToTensor() ]) 

        #self.transform = transforms.Compose([transforms.PILToTensor()])

        _, self.preprocess = clip.load("ViT-B/32", device="cuda")

        self.device = "cuda"
        #self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


    def __getitem__(self, idx):
        item = self.dataset[idx]
        bold_path = numpy.load (self.dataset[idx]["bold_signal"])
        bold_signal = torch.Tensor (bold_path)#.unsqueeze (0)
        caption = self.prompt + self.pre_caption(self.dataset[idx]["text-output"])
        query = self.pre_caption(self.dataset[idx]["text-input"])

        image = Image.open(self.dataset[idx]["image"])
        image = image.resize((224,224))

        images_input = self.preprocess(image)

        return {"bold_signal": bold_signal, "text_input":query, "text_output":caption, "image": images_input}

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict

def data_builder (batch_size=16):
    with open("data/train.json") as json_file:
        train_dataset = ImageCaptioningDataset (json.load(json_file))


    with open("data/test.json") as json_file:
        test_dataset = ImageCaptioningDataset (json.load(json_file))


    # with open("data/valid.json") as json_file:
    #     valid_dataset = ImageCaptioningDataset (json.load(json_file))


    data_loaders = {}

    data_loaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_loaders["test"]  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    #data_loaders["valid"] = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return data_loaders
