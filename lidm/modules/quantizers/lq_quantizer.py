import torch
import torch.nn.functional as F
from einops import rearrange

from .base_quantizer import BaseQuantizer
from .utils import quantize

class LQQuantizer(BaseQuantizer):

    def __init__(self, num_latents, embedding_dim, init_type, quantization, commitment):
        super(LQQuantizer, self).__init__(num_latents, embedding_dim, init_type)

        self.quantization = quantization
        self.commitment = commitment
    
    def forward(self, z):
        

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.num_latents, 1)

        z_q, indices = quantize(z_flattened, self.codebook.weight)

        quantized_sg = z_flattened + (z_q - z_flattened).detach()

        quantized_sg = quantized_sg.view(z.shape)
        quantized_sg = rearrange(quantized_sg, 'b h w c -> b c h w').contiguous()


        quantization_loss = self.quantization_loss(z_flattened, z_q)

        commitment_loss = self.commitment_loss(z_flattened, z_q)

        loss = self.quantization*quantization_loss + self.commitment*commitment_loss

        return quantized_sg, loss
    
    

    def quantization_loss(self, continuous, quantized):
        return torch.mean(torch.square(torch.clone(continuous).detach() - quantized))

    def commitment_loss(self, continuous, quantized):
        return torch.mean(torch.square(continuous - torch.clone(quantized).detach()))

    
    def get_codebook_entry(self, z, shape):

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.num_latents, 1)

        z_q, indices = quantize(z_flattened, self.codebook.weight)

        quantized_sg = z_flattened + (z_q - z_flattened).detach()

        quantized_sg = quantized_sg.view(z.shape)
        quantized_sg = rearrange(quantized_sg, 'b h w c -> b c h w').contiguous()
        return quantized_sg