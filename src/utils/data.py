from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              random_split)

from .PetsDataset import PetsDataset


def create_train_test_loaders(batch_size=32) -> tuple[DataLoader, DataLoader]:
    data = PetsDataset()

    train_data, test_data = random_split(data, (0.8, 0.2))

    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

    return train_loader, test_loader


if __name__ == "__main__":
    dataloader = create_train_test_loaders()
    size, image, mask = next(iter(dataloader))
    print(image)
