import torch
from torch import nn;
from torch.nn import functional as f
from decoder import vae_attention,vae_residual
#model reduce the dim. of the data increase no. of feature
class vae_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(batch_size,channel,ht,width)->(batch_size,128,ht,width)
            nn.Conv2d(3,128,kernel_size=3,padding=1),
             #(batch_size,128,ht,width)->(batch_size,128,ht,width)
             vae_residual(128,128),#residual combination of conv. and norm.
             #(batch_size,128,ht,width)->(batch_size,128,ht,width)
             vae_residual(128,128),
            #(batch_size,128,ht,width)->(batch_size,128,ht/2,width/2)
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            #(batch_size,128,ht/2,width/2)->(batch_size,256,ht/2,width/2)
             vae_residual(128,256),
            #(batch_size,256,ht/2,width/2)->(batch_size,256,ht/2,width/2)
             vae_residual(256,256),
             #(batch_size,256,ht/2,width/2)->(batch_size,256,ht/4,width/4)
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
            #(batch_size,256,ht/4,width/4)->(batch_size,512,ht/4,width/4)
             vae_residual(256,512),
            #(batch_size,512,ht/4,width/4)->(batch_size,512,ht/4,width/4)
             vae_residual(512,512),
            #(batch_size,512,ht/4,width/4)->(batch_size,512,ht/8,width/8)
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
            vae_residual(512,512),
            vae_residual(512,512),
            #(batch_size,512,ht/8,width/8)->-(batch_size,512,ht/8,width/8)
            vae_residual(512,512),
             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            vae_attention(512),#self attention over each pixel image made of pixel attention way to relate pixel to each
                                #other
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            vae_residual(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        def forward(self, x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
            # x: (Batch_Size, Channel, Height, Width)
            # noise: (Batch_Size, 4, Height / 8, Width / 8)
            for module in self:

                if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                    x = f.pad(x, (0, 1, 0, 1))#here apply padding on which part
            
                x = module(x)
                # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
                mean, log_variance = torch.chunk(x, 2, dim=1)
                # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
                log_variance = torch.clamp(log_variance, -30, 20)
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
                variance = log_variance.exp()
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
                stdev = variance.sqrt()
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
                x = mean + stdev * noise
        
                # Scale by a constant
                x *= 0.18215
        
                return x
