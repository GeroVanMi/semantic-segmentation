import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize


def create_dataloader() -> DataLoader:
    return DataLoader(PetsDataset(), batch_size=2, shuffle=True)


def extract_height(img_data):
    return img_data["size"][0]


def extract_width(img_data):
    return img_data["size"][1]


def read_json_file(file_path):
    df = pd.read_json(file_path).T
    df["height"] = df["imgdata"].apply(extract_height)
    df["width"] = df["imgdata"].apply(extract_width)
    return df


def get_mask(path):
    mask = read_image(path).to(torch.int64)
    mask = torch.nn.functional.one_hot(mask, 3).to(torch.float32)
    mask = tv_tensors.Mask(mask.permute(3, 1, 2, 0).squeeze(3))
    return mask


class PetsDataset(Dataset):
    def __init__(self, data_folder_path="../data", resize_to=(256, 256)):
        self.data = read_json_file(f"{data_folder_path}/data.json")
        self.data_folder_path = data_folder_path
        self.resize_to = resize_to

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.index[idx]
        image_size = self.data[["height", "width"]].iloc[idx]

        image = read_image(f"{self.data_folder_path}/images/{image_name}.png")
        mask = get_mask(f"{self.data_folder_path}/masks/{image_name}.png")

        resize = Resize(self.resize_to, antialias=True)
        return [image_size.to_numpy(), resize(image), resize(mask)]


if __name__ == "__main__":
    dataloader = create_dataloader()
    size, image, mask = next(iter(dataloader))
    print(image)
