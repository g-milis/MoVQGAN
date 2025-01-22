import torch


# import random

# def code_to_cluster(clusters):
#     """
#     Create a dictionary to map each code to its corresponding cluster.
#     """
#     code_to_cluster = {}
#     for cluster_id, codes in clusters.items():
#         for code in codes:
#             code_to_cluster[code.item()] = cluster_id
#     return code_to_cluster

# def replace_codes_within_clusters(code_sequence, clusters, rate):
#     """
#     Replace each code in the code_sequence with a random code from the same cluster.
#     """
#     code_to_cluster_dict = code_to_cluster(clusters)

#     modified_sequence = []

#     # Randomly select a different code
#     for code in code_sequence:
#         if random.random() < rate:
#             cluster_id = code_to_cluster_dict[code.item()]
#             new_code = random.choice(clusters[cluster_id])
#             modified_sequence.append(new_code)
#         else:
#             modified_sequence.append(code)

#     return torch.tensor(
#         modified_sequence,
#         dtype=code_sequence.dtype,
#         device=code_sequence.device
#     )


def code_to_cluster(clusters):
    """
    Create a tensor-based mapping from each code to its corresponding cluster.
    """
    max_code = max(max(codes).item() for codes in clusters.values()) + 1
    code_to_cluster_tensor = torch.full((max_code,), -1, device=next(iter(clusters.values())).device)

    for cluster_id, codes in clusters.items():
        code_to_cluster_tensor[codes] = cluster_id

    return code_to_cluster_tensor


def replace_codes_within_clusters(code_sequence, clusters, rate):
    """
    Replace each code in the code_sequence with a random code from the same cluster.
    """
    code_to_cluster_tensor = code_to_cluster(clusters)

    modified_sequence = code_sequence.clone()

    # Generate random replacement mask
    random_values = torch.rand(len(code_sequence), device=code_sequence.device)
    replace_mask = random_values < rate
    # Get the indices where the mask is True
    replace_indices = torch.nonzero(replace_mask, as_tuple=True)[0]

    for idx in replace_indices:
        cluster_id = code_to_cluster_tensor[code_sequence[idx]].item()
        codes_in_cluster = clusters[cluster_id]
        new_code = codes_in_cluster[torch.randint(len(codes_in_cluster), (1,), device=code_sequence.device)]
        modified_sequence[idx] = new_code

    return modified_sequence


def get_conditional_codes(key, codes):
    """
    Use the key to seed the shuffling, then select the first two codes.
    """
    torch.manual_seed(key)
    indices = torch.randperm(len(codes), device=codes.device)
    return codes[indices[0]], codes[indices[1]]


def redgreen_embed_payload(code_sequence, clusters=None, payload=None, n_payload_bits=16):
    """
    Replace each code in the code_sequence another code from the same cluster based on the previous code's hash and the payload bits.
    """
    if clusters is None or payload is None:
        return code_sequence

    assert isinstance(payload, int)

    code_to_cluster_tensor = code_to_cluster(clusters)
    modified_sequence = torch.empty_like(code_sequence, device=code_to_cluster_tensor.device)

    binary_bits = torch.tensor(list(map(int, bin(payload)[2:].zfill(n_payload_bits))), dtype=torch.int32)
    step_size = len(code_sequence) // n_payload_bits

    for bit_index in range(n_payload_bits):
        bit = binary_bits[bit_index]
        start_idx = bit_index * step_size
        end_idx = min(start_idx + step_size, len(code_sequence))

        for code_index in range(start_idx, end_idx):
            cluster_id = code_to_cluster_tensor[code_sequence[code_index].item()].item()
            codes_in_cluster = clusters[cluster_id]

            if len(codes_in_cluster) > 1 and code_index > 0:
                code_0, code_1 = get_conditional_codes(
                    modified_sequence[code_index - 1].item(),
                    codes_in_cluster
                )
            else:
                code_0 = code_sequence[code_index].item()
                code_1 = code_sequence[code_index].item()

            if bit == 0:
                modified_sequence[code_index] = code_0
            else:
                modified_sequence[code_index] = code_1

    return modified_sequence


def detect_payload(code_sequence, clusters, n_payload_bits=16, threshold=0.1):
    """
    Detect the payload from the code_sequence based on the previous code's hash.
    """
    code_to_cluster_tensor = code_to_cluster(clusters)

    detections = torch.full((n_payload_bits, len(code_sequence)), float('nan'), device=code_to_cluster_tensor.device)

    step_size = len(code_sequence) // n_payload_bits
    for bit_index in range(n_payload_bits):
        start_idx = bit_index * step_size
        end_idx = start_idx + step_size
        for code_index in range(start_idx, min(end_idx, len(code_sequence))):
            if code_index == 0:
                continue

            cluster_id = code_to_cluster_tensor[code_sequence[code_index].item()].item()
            codes_in_cluster = clusters[cluster_id]

            if len(codes_in_cluster) > 1:
                code_0, code_1 = get_conditional_codes(
                    code_sequence[code_index - 1].item(),
                    codes_in_cluster
                )

                if code_sequence[code_index] == code_0:
                    detections[bit_index, code_index] = 0
                elif code_sequence[code_index] == code_1:
                    detections[bit_index, code_index] = 1

    average_bits = torch.nanmean(detections, dim=1)
    
    # Check if the average bits are sufficiently close to either 0 or 1
    if torch.any(torch.abs(0.5 - average_bits) < 0.5 - threshold):
        payload = None
    else:
        payload_bits = (average_bits > 0.5).int()
        payload = int("".join(map(str, payload_bits.tolist())), 2)

    return {
        "watermarked": payload is not None,
        "payload": payload,
        "average_bits": average_bits.tolist()
    }
