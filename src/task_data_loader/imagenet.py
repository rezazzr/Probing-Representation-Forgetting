import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        normalize,
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

mini_train_transform = transforms.Compose(
    [
        transforms.Resize(110),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        normalize,
    ]
)

mini_valid_transform = transforms.Compose(
    [
        transforms.Resize(110),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ]
)
