import torch
import torch.nn.functional as F


def quantize(z, codebook):
    # find distances between the embeddings and the codebook
    codebook = torch.clone(codebook).unsqueeze(0)
    d = torch.abs(z-codebook)
    indices = torch.argmin(d, dim=-1, keepdim=True)
    z_q = torch.repeat_interleave(codebook, repeats=z.shape[0], dim=0)
    z_q = torch.take_along_dim(z_q, indices, dim=2)
    return z_q, indices

def find_min_indices(d):
    # find closest indices
    min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

    return min_encoding_indices

def measure_perplexity(encodings):
    # measure perplexity
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()

    return perplexity
