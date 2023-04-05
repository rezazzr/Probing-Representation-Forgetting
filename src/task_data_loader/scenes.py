import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Scenes(Dataset):
    target_str = [
        "kitchen",
        "operating_room",
        "restaurant_kitchen",
        "videostore",
        "poolinside",
        "mall",
        "kindergarden",
        "buffet",
        "hospitalroom",
        "library",
        "inside_bus",
        "bar",
        "dentaloffice",
        "office",
        "computerroom",
        "grocerystore",
        "cloister",
        "concert_hall",
        "jewelleryshop",
        "laundromat",
        "warehouse",
        "gym",
        "lobby",
        "meeting_room",
        "garage",
        "inside_subway",
        "restaurant",
        "children_room",
        "corridor",
        "hairsalon",
        "bookstore",
        "movietheater",
        "elevator",
        "stairscase",
        "artstudio",
        "bathroom",
        "gameroom",
        "locker_room",
        "nursery",
        "waitingroom",
        "winecellar",
        "florist",
        "closet",
        "clothingstore",
        "pantry",
        "prisoncell",
        "shoeshop",
        "museum",
        "fastfood_restaurant",
        "auditorium",
        "subway",
        "classroom",
        "laboratorywet",
        "deli",
        "tv_studio",
        "bedroom",
        "bowling",
        "livingroom",
        "dining_room",
        "trainstation",
        "airport_inside",
        "church_inside",
        "toystore",
        "casino",
        "bakery",
        "greenhouse",
        "studiomusic",
    ]
    target_str_to_id = {t: index for index, t in enumerate(target_str)}

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.join(os.path.expanduser(root), "Scenes")
        self.transform = transform
        self.loader = loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

    def _load_metadata(self) -> None:

        if self.train:
            self.data = pd.read_csv(
                filepath_or_buffer=os.path.join(self.root, "TrainImages.txt"),
                sep="/",
                names=["class_name", "image_name"],
            )
        else:
            self.data = pd.read_csv(
                filepath_or_buffer=os.path.join(self.root, "TestImages.txt"),
                sep="/",
                names=["class_name", "image_name"],
            )
        self.data["target_id"] = self.data["class_name"].apply(lambda c_name: self.target_str_to_id[c_name])

    def _check_integrity(self) -> bool:
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, row["class_name"], row["image_name"])
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample["class_name"], sample["image_name"])
        target = sample["target_id"]  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
