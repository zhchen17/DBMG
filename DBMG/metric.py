import torch
import numpy as np

def compute_ap(ranked_labels, query_label):
    relevant = (ranked_labels == query_label).int()
    n_relevant = relevant.sum().item()
    if n_relevant == 0:
        return 0.0
    precisions = torch.cumsum(relevant, dim=0).float() / (torch.arange(len(relevant), device=relevant.device).float() + 1)
    return (precisions * relevant.float()).sum().item() / n_relevant

def compute_map(similarity_matrix, text_labels, image_labels):
    num_texts = similarity_matrix.shape[0]
    APs = []
    for i in range(num_texts):
        ranked_idx = np.argsort(-similarity_matrix[i].cpu().detach().numpy())
        ranked_image_labels = image_labels[ranked_idx]
        APs.append(compute_ap(ranked_image_labels, text_labels[i]))
    return np.mean(APs)