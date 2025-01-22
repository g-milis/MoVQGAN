import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, l1_loss

from dataset import JpegDataset
from movqgan import get_movqgan_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def postprocess(image_tensor):
    image_tensor = image_tensor.permute(1, 2, 0).numpy()
    image_tensor = ((image_tensor + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return image_tensor


def get_cluster_indices(kmeans):
    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx)
    return clusters


# SETTINGS
k_list = [100, 1000, 5000]
rate_list = [0.1, 0.5, 0.75]
device = "mps"
batch_size = 4


model = get_movqgan_model('67M', pretrained=True, device=device)

embeddings = model.quantize.embedding.weight
embeddings = embeddings.detach().cpu().numpy()

mse_results = []
l1_results = []

for k in k_list:
    mse_row = []
    l1_row = []

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    clusters = get_cluster_indices(kmeans)

    for rate in rate_list:

        print(f"K={k}, Rate={rate}")

        dataloader = DataLoader(
            JpegDataset(
                "/Users/georgemilis/Research/Small-ImageNet-Validation-Dataset-1000-Classes"
            ),
            batch_size=batch_size,
            shuffle=True
        )

        total_mse = 0.0
        total_l1 = 0.0
        total_samples = 0

        for batch_images in dataloader:
            batch_images = batch_images.to(device)

            with torch.no_grad():
                quant, loss, (perplexity, min_encodings, codebook_indices) = model.encode(batch_images, clusters=clusters, rate=rate)
                out = model.decode(quant)

            batch_mse = mse_loss(batch_images, out).item()
            batch_l1 = l1_loss(batch_images, out).item()

            # Accumulate loss
            total_mse += batch_mse * batch_images.size(0)
            total_l1 += batch_l1 * batch_images.size(0)
            total_samples += batch_images.size(0)

            # # Debug ===========================================================
            # from PIL import Image
            # image_tensor_1 = batch_images.cpu()[0]
            # image_tensor_2 = out.cpu()[0]
            # image1 = postprocess(image_tensor_1)
            # image2 = postprocess(image_tensor_2)
            # image1 = Image.fromarray(image1)
            # image2 = Image.fromarray(image2)
            # image1.save("sample_1.jpeg")
            # image2.save("sample_2.jpeg")
            # raise


        # Compute the average losses for the dataset
        avg_mse = total_mse / total_samples
        avg_l1 = total_l1 / total_samples

        print(f"Average MSE: {avg_mse}")
        print(f"Average L1: {avg_l1}")

        mse_row.append(avg_mse)
        l1_row.append(avg_l1)

    mse_results.append(mse_row)
    l1_results.append(l1_row)


# Create DataFrames for MSE and L1
mse_df = pd.DataFrame(mse_results, index=k_list, columns=rate_list)
l1_df = pd.DataFrame(l1_results, index=k_list, columns=rate_list)
mse_df.index.name = 'K'
mse_df.columns.name = 'Rate'
l1_df.index.name = 'K'
l1_df.columns.name = 'Rate'

print("MSE Results:")
print(mse_df.to_string())
print("\nL1 Results:")
print(l1_df.to_string())
