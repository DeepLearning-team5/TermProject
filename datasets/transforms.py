from torchvision import transforms

def get_transform(train):
    transforms =[]
    transforms.append(transforms.ToTensor())
    if train:
        transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms)