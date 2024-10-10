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
    def __init__(self, autoencoder1, autoencoder2, device):
        super(JointAutoencoder, self).__init__()
        self.ae1 = autoencoder1
        self.ae2 = autoencoder2
        self.device = device
    
    def encode1(self, x): 
        return self.ae1.encode(x)

    def encode2(self, x):
        return self.ae2.encode(x)

    def decode1(self, x):
        return self.ae1.decode(x)

    def decode2(self, x):
        return self.ae2.decode(x)
        
    def forward(self, x):
        att_in, weight_in, _ = x  
        att_in = att_in.to(self.device)  
        weight_in = weight_in.to(self.device) 

        latent_att = self.encode1(att_in)
        latent_weight = self.encode2(weight_in)
        
        att_from_att = self.decode1(latent_att)
        att_from_weight = self.decode1(latent_weight)
        weight_from_weight = self.decode2(latent_weight)
        weight_from_att = self.decode2(latent_att)
        
        return att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight
