from transformers import AutoModelForSemanticSegmentation
import torch
import torch.nn as nn

class DirectSAMFeaturizer(nn.Module):
    def __init__(self, checkpoint="chendelong/DirectSAM-1800px-0424", feature_size='14x14'):
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self.feature_size = feature_size

    def forward(self, x):
        with torch.no_grad():
            encoder_outputs = self.model.segformer.encoder(
                x, 
                output_hidden_states=True, 
                return_dict=True
            )
            
            hidden_states = encoder_outputs.hidden_states
            
            if self.feature_size == '56x56':
                return hidden_states[0]
            elif self.feature_size == '28x28':
                return hidden_states[1]
            elif self.feature_size == '14x14':
                return hidden_states[2]
            elif self.feature_size == '7x7':
                return hidden_states[3]
            else:
                raise ValueError(f"Unsupported feature size: {self.feature_size}")

def get_direct_sam(feature_size='14x14'):
    model = DirectSAMFeaturizer(feature_size=feature_size)
    
    if feature_size == '56x56':
        patch_size = 4
        dim = 64
    elif feature_size == '28x28':
        patch_size = 7
        dim = 128
    elif feature_size == '14x14':
        patch_size = 14
        dim = 320
    elif feature_size == '7x7':
        patch_size = 14
        dim = 512
    else:
        raise ValueError(f"Unsupported feature size: {feature_size}")
    
    return model, patch_size, dim

def get_feature_dimensions(model, sample_input):
    with torch.no_grad():
        output = model(sample_input)
    return output.shape[1]  # Assuming output shape is [B, C, H, W]