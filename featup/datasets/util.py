import os
from torch.utils.data import Dataset
from featup.datasets.ImageNetSubset import ImageNetSubset
from featup.datasets.COCO import Coco
from featup.datasets.DAVIS import DAVIS
from featup.datasets.SampleImage import SampleImage
from featup.datasets.dataset_wbc import DatasetMarr


class SlicedDataset(Dataset):
    def __init__(self, ds, start, end):
        self.ds = ds
        self.start = max(0, start)
        self.end = min(len(ds), end)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration

        return self.ds[self.start + index]

    def __len__(self):
        return self.end - self.start



class SingleImageDataset(Dataset):
    def __init__(self, i, ds, l=None):
        self.ds = ds
        self.i = i
        self.l = len(self.ds) if l is None else l

    def __len__(self):
        return self.l

    def __getitem__(self, item):
        return self.ds[self.i]


def get_dataset(dataroot, name, split, transform, target_transform, include_labels):
    if name == 'imagenet':
        if split == 'val':
            imagenet_subset = f'datalists/val_paths_vit.txt'
        else:
            imagenet_subset = None

        return ImageNetSubset(dataroot, split, transform, target_transform,
                              include_labels=include_labels, subset=imagenet_subset)
    elif name == 'cocostuff':
        return Coco(dataroot, split, transform, target_transform, include_labels=include_labels)
    elif name.startswith('davis_'):
        return DAVIS(dataroot, name.split("_")[-1], transform)
    elif name == "sample":
        return SampleImage(
            paths=["../sample-images/bird_left.jpg",
                   "../sample-images/bird_right.jpg"],
            transform=transform
        )
    elif name == "marr":
        dataset_selection = ["Ace_20", "Mat_19", "MLL_20", "BMC_22"]  # Modify as needed
        labels_map = {
            'basophil': 0,
            'eosinophil': 1,
            'erythroblast': 2,
            'promyelocyte': 3,
            'myelocyte': 4,
            'metamyelocyte': 5,
            'neutrophil_banded': 6,
            'neutrophil_segmented': 7,
            'monocyte': 8,
            'lymphocyte_typical': 9
        }
        fold = 0  # Adjust according to your needs
        is_hsv = False  # Adjust according to your needs
        is_hed = False  # Adjust according to your needs
        return DatasetMarr(dataroot, dataset_selection, labels_map, fold, transform, split, is_hsv, is_hed)
    elif name == "custom":
        list_img = os.listdir(dataroot)
        return SampleImage(
            paths=[os.path.join(dataroot, path) for path in list_img],
            transform=transform
        ) 
    else:
        raise ValueError(f"Unknown dataset {name}")
