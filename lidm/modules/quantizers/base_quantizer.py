from torch import nn

class BaseQuantizer(nn.Module):
    def __init__(self, num_latents, embedding_dim, init_type):
        super(BaseQuantizer, self).__init__()

        self.embedding_dim = embedding_dim

        self.num_latents = num_latents
        
        self.codebook = nn.Embedding(self.num_latents, self.embedding_dim)
        if init_type == "uniform": # otherwise, it is initialized with normal distribution
            self.codebook.weight.data.uniform_(-1.0 / self.embedding_dim, 
                                                1.0 / self.embedding_dim)
    def forward(self, z):
        raise NotImplementedError()
