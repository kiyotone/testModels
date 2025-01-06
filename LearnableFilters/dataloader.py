from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
import albumentations as A

DATASET_PATH ="D:/Datasets/SuperResolution/"

HIGH_RES = 96
LOW_RES = HIGH_RES // 4

highres_transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transforms = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        
        for label, class_name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, class_name))[:30]
            self.data += list(zip(files, [label]*len(files)))
            
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        
        img = Image.open(os.path.join(root_and_dir, img_file)).convert("RGB")  # Convert to RGB
        img = np.array(img)
        
        # Apply both transforms
        img = both_transforms(image=img)["image"]
        
        # Apply high-res and low-res transforms
        high_res = highres_transforms(image=img)["image"]
        low_res = lowres_transforms(image=img)["image"]
        
        return high_res, low_res

def test():
    dataset = MyImageFolder(DATASET_PATH + "dataset/raw_data")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        save_image(x, "high_res.png")
        save_image(y, "low_res.png")
        

if __name__ == "__main__":
    test()