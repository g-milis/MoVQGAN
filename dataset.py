import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class JpegDataset(Dataset):
    """
    A PyTorch dataset that traverses a directory and its subdirectories
    to find .jpeg files and allows access to them in batches.
    """
    def __init__(self, root_dir, max_items=None):
        self.root_dir = root_dir
        self.transform = T.Compose([
            T.RandomResizedCrop(512, scale=(1., 1.), ratio=(1., 1.), interpolation=T.InterpolationMode.BICUBIC),
        ])
        self.file_paths = self._find_jpeg_files(root_dir)

        if max_items is not None:
            self.file_paths = self.file_paths[:max_items]


    def preprocess(self, image):
        """Image to Tensor."""
        image = self.transform(image).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32) / 127.5 - 1
        image_tensor = torch.from_numpy(np.transpose(image, [2, 0, 1]))
        return image_tensor


    def postprocess(self, image_tensor):
        """Tensor to Image."""
        image_tensor = image_tensor.permute(1, 2, 0).numpy()
        image_tensor = ((image_tensor + 1) * 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_tensor)
        return image


    def _find_jpeg_files(self, directory):
        jpeg_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".jpeg"):
                    jpeg_files.append(os.path.join(root, file))
        return jpeg_files


    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path)
        image_tensor = self.preprocess(image)
        return img_path, image_tensor


if __name__ == "__main__":
    dataset = JpegDataset(root_dir="/Users/georgemilis/Research/Small-ImageNet-Validation-Dataset-1000-Classes")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for batch_images in dataloader:
        print(f"Batch images shape: {batch_images.shape}")

        image_tensor = batch_images[0].cpu()
        image_tensor = image_tensor.permute(1, 2, 0).numpy()
        image_tensor = ((image_tensor + 1) * 127.5).clip(0, 255).astype(np.uint8)

        image = Image.fromarray(image_tensor)
        image.save("sample.jpeg")

        break
