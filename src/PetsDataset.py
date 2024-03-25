import json

import pandas as pd
from torch.utils.data import DataLoader, Dataset


def create_dataloader() -> DataLoader:
    return DataLoader(PetsDataset())


def extract_height(img_data):
    return img_data["size"][0]


def extract_width(img_data):
    return img_data["size"][1]


def read_json_file(file_path):
    df = pd.read_json(file_path).T
    df["height"] = df["imgdata"].apply(extract_height)
    df["width"] = df["imgdata"].apply(extract_width)
    return df


class PetsDataset(Dataset):
    def __init__(self, data_file_path="../data/data.json"):
        self._data = read_json_file(data_file_path)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[["height", "width"]].iloc[idx]
        return item.to_numpy()


if __name__ == "__main__":
    dataloader = create_dataloader()
    for entry in dataloader:
        print(entry)
