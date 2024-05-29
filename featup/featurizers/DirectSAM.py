import torch
import torch.nn as nn
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from torchsummary import summary

class DirectSAMFeaturizer(nn.Module):

    def __init__(self, checkpoint="chendelong/DirectSAM-1800px-0424"):
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
        # self.model.eval()  # Set model to evaluation mode
        # print summary of the model


    def forward(self, img):
        # Process the image using the image processor
        inputs = self.image_processor(img, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Get the features from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Extract the features
        features = outputs.logits  # Assuming logits are the features we want
        return features
