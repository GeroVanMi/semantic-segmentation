import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


def create_dataloader() -> DataLoader:
    return DataLoader(PetsDataset(), batch_size=32, shuffle=True)


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
    def __init__(self, data_folder_path="../data"):
        self.data = read_json_file(f"{data_folder_path}/data.json")
        self.data_folder_path = data_folder_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.index[idx]
        image = read_image(f"{self.data_folder_path}/images/{name}.png")
        item = self.data[["height", "width"]].iloc[idx]
        # TODO: Add the mask matrix as well.
        return {
            "size": item.to_numpy(),
            "image": image,
        }


if __name__ == "__main__":
    dataloader = create_dataloader()
    items = next(iter(dataloader))
    print(items)
