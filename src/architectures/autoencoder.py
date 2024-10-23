import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, output_dim=None, num_layers=3, vae=False, bias=True):
        super(Autoencoder, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim   
        self.embed_dim = [2 * embed_dim, embed_dim] if vae else [embed_dim, embed_dim]

        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], self.output_dim)
            )
        elif num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )
        elif num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )

    def encode(self, x):
        return self.encoder(x)  # Assumendo che x sia già sulla GPU

    def decode(self, x):
        return self.decoder(x)  # Assumendo che x sia già sulla GPU

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)
    
class JointAutoencoder(nn.Module):
    def __init__(self, autoencoder_descr, autoencoder_weight, device):
        super(JointAutoencoder, self).__init__()
        self.ae_d = autoencoder_descr
        self.ae_w = autoencoder_weight
        self.device = device
    
    def encode_descr(self, x): 
        return self.ae_d.encode(x)

    def encode_weight(self, x):
        return self.ae_w.encode(x)

    def decode_descr(self, x):
        return self.ae_d.decode(x)

    def decode_weight(self, x):
        return self.ae_w.decode(x)
        
    def forward(self, x):
        descr_in, weight_in, _ = x  
        descr_in = descr_in.to(self.device)  
        weight_in = weight_in.to(self.device) 

        latent_descr = self.encode_descr(descr_in)
        latent_weight = self.encode_weight(weight_in)
        
        descr_from_descr = self.decode_descr(latent_descr)
        descr_from_weight = self.decode_descr(latent_weight)
        weight_from_weight = self.decode_weight(latent_weight)
        weight_from_descr = self.decode_weight(latent_descr)
        
        return descr_from_descr, descr_from_weight, weight_from_weight, weight_from_descr, latent_descr, latent_weight
