import os
import torch
from sklearn.cluster import KMeans
from PIL import Image

from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, l1_loss

import watermarking
from dataset import JpegDataset
from movqgan import get_movqgan_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_cluster_indices(clustering_object):
    clusters = {i: [] for i in range(clustering_object.n_clusters)}
    for idx, label in enumerate(clustering_object.labels_):
        clusters[label].append(idx)
    clusters = {k: torch.tensor(v, device=device) for k, v in clusters.items()}
    return clusters


# SETTINGS
k = 5000
model_size = "270M" # [67M, 102M, 270M]
batch_size = 1
device = "mps"
payload = 2025
n_payload_bits = 32


for model_size in ["67M", "102M", "270M"]:

    print(model_size)

    model = get_movqgan_model(model_size, pretrained=True, device=device)

    embeddings = model.quantize.embedding.weight
    embeddings = embeddings.detach().cpu().numpy()

    mse_results = []
    l1_results = []


    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    clusters = get_cluster_indices(kmeans)

    dataset = JpegDataset(
        "/Users/georgemilis/Research/Small-ImageNet-Validation-Dataset-1000-Classes",
        max_items=50
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    total_mse = 0.0
    total_l1 = 0.0
    total_samples = 0
    detections = 0

    for batch_paths, batch_images in dataloader:

        batch_images = batch_images.to(device)

        with torch.no_grad():
            quant, loss, (perplexity, min_encodings, codebook_indices) = model.encode(batch_images, clusters=clusters)

            codes = codebook_indices
            watermarked_indices = watermarking.redgreen_embed_payload(codes, clusters=clusters, payload=payload, n_payload_bits=n_payload_bits)
            watermarked_image = model.decode_code(watermarked_indices.view(1, 64, 64))

            save_path = f"/Users/georgemilis/Research/MoVQGAN/out/{os.path.basename(batch_paths[0]).replace(".JPEG", "")}_watermarked_{k}_{model_size}.png"
            watermarked_image_tensor = watermarked_image.cpu()[0]
            watermarked_image = dataset.postprocess(watermarked_image_tensor)
            watermarked_image.save(save_path)

            watermarked_image = Image.open(save_path)
            watermarked_image_tensor = dataset.preprocess(watermarked_image).to(device).unsqueeze(0)

            quant, loss, (perplexity, min_encodings, reencoded_indices) = model.encode(watermarked_image_tensor)

            print("Code preservation:", (watermarked_indices == reencoded_indices).cpu().numpy().mean())

            detection_info = watermarking.detect_payload(reencoded_indices, clusters=clusters, n_payload_bits=n_payload_bits)
            if detection_info["watermarked"]: # always True
                if detection_info["payload"] == payload:
                    detections += 1

        batch_mse = mse_loss(batch_images, watermarked_image_tensor).item()
        batch_l1 = l1_loss(batch_images, watermarked_image_tensor).item()

        # Accumulate loss
        total_mse += batch_mse * batch_images.size(0)
        total_l1 += batch_l1 * batch_images.size(0)
        total_samples += batch_images.size(0)

        # break

    # Compute the average losses for the dataset
    avg_mse = total_mse / total_samples
    avg_l1 = total_l1 / total_samples
    avg_detection = detections / total_samples

    print(f"Average MSE: {avg_mse}")
    print(f"Average L1: {avg_l1}")
    print(f"Average Detection: {avg_detection}")
