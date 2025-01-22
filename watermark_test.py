import torch
import numpy as np
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, l1_loss

import watermarking
from dataset import JpegDataset
from movqgan import get_movqgan_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def postprocess(image_tensor):
    image_tensor = image_tensor.permute(1, 2, 0).numpy()
    image_tensor = ((image_tensor + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return image_tensor


def get_cluster_indices(clustering_object):
    clusters = {i: [] for i in range(clustering_object.n_clusters)}
    for idx, label in enumerate(clustering_object.labels_):
        clusters[label].append(idx)
    clusters = {k: torch.tensor(v, device=device) for k, v in clusters.items()}
    return clusters


# SETTINGS
k = 1000
model_size = "270M" # [67M, 102M, 270M]
batch_size = 1
device = "mps"
payload = 0b1010101010101010101010101010101010101010101010101010101010101000


model = get_movqgan_model(model_size, pretrained=True, device=device)

embeddings = model.quantize.embedding.weight
embeddings = embeddings.detach().cpu().numpy()

mse_results = []
l1_results = []


kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(embeddings)
clusters = get_cluster_indices(kmeans)

dataloader = DataLoader(
    JpegDataset(
        "/Users/georgemilis/Research/Small-ImageNet-Validation-Dataset-1000-Classes",
        max_items=5
    ),
    batch_size=batch_size,
    shuffle=False
)

total_mse = 0.0
total_l1 = 0.0
total_samples = 0
detections = 0

for idx, batch_images in enumerate(dataloader):

    batch_images = batch_images.to(device)

    with torch.no_grad():
        quant, loss, (perplexity, min_encodings, codebook_indices) = model.encode(batch_images, clusters=clusters)

        codes = codebook_indices
        watermarked = watermarking.redgreen_embed_payload(codes, clusters=clusters, payload=payload, n_payload_bits=64)

        detection_info = watermarking.detect_payload(watermarked, clusters=clusters, n_payload_bits=64)
        if detection_info["watermarked"]:
            if detection_info["payload"] == payload:
                detections += 1
        print(detection_info["average_bits"])

        out = model.decode_code(watermarked.view(batch_size, 64, 64))


    batch_mse = mse_loss(batch_images, out).item()
    batch_l1 = l1_loss(batch_images, out).item()

    # Accumulate loss
    total_mse += batch_mse * batch_images.size(0)
    total_l1 += batch_l1 * batch_images.size(0)
    total_samples += batch_images.size(0)


# Compute the average losses for the dataset
avg_mse = total_mse / total_samples
avg_l1 = total_l1 / total_samples
avg_detection = detections / total_samples

print(f"Average MSE: {avg_mse}")
print(f"Average L1: {avg_l1}")
print(f"Average Detection: {avg_detection}")
