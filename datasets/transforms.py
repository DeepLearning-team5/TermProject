from torchvision import transforms

def get_transform(train):
    transforms_list =[]
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Resize((224,224)))
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)