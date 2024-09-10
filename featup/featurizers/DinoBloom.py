import torch
import torch.nn as nn
import os
import urllib.request

class DinoBloom(nn.Module):
    def __init__(self, model_name="dinov2_vits14", patch_size=14, activation_type="key"):
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.activation_type = activation_type
        self.embed_size = 384  # for dinov2_vits14
        
        # Define the model path
        self.model_path = "dinobloom-s.pth"
        
        # Download the model if it doesn't exist
        self.download_model_if_needed()
        
        # Load the original DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        
        # Load finetuned weights
        pretrained = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        # Make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        
        # Update position embedding for 224x224 image (16x16 patches)
        pos_embed = nn.Parameter(torch.zeros(1, 257, self.embed_size))
        self.model.pos_embed = pos_embed
        
        # Load the new state dict
        self.model.load_state_dict(new_state_dict, strict=True)
        
        # Add a convolution layer to adjust the spatial dimensions if needed
        self.adjust_spatial = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=3, padding=1, stride=2)
    
    def download_model_if_needed(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading DinoBloom model to {self.model_path}...")
            url = "https://zenodo.org/records/10908163/files/DinoBloom-S.pth?download=1"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")
        else:
            print(f"DinoBloom model already exists at {self.model_path}")

    def forward(self, x):
        with torch.no_grad():
            features_dict = self.model.forward_features(x)
            features = features_dict['x_norm_patchtokens']
        
        # Reshape features to [B, C, H, W] format
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Adjust spatial dimensions if needed
        if H != 16:  # Assuming the expected size is 16x16
            features = self.adjust_spatial(features)
        
        return features