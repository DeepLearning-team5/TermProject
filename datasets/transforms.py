from torchvision import transforms

def get_transform(train):
    transforms_list =[]
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Resize((320,320)))
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        #transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    return transforms.Compose(transforms_list)