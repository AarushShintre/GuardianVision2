class SemiSupervisedDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.keys = list(data.keys())
        self.actions = [key.split("instance_")[0] for key in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        images, center = self.data[key]
        action = self.actions[idx]

        if self.transform:
            images = [self.transform(image) for image in images]
        images = torch.stack(images)
        return images, torch.tensor(center, dtype=torch.float32), action

