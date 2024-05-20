import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from bert import BertModel
from utils import *

class Discriminator(nn.Module):
    def __init__(self, d_hidden_size, dkp, num_labels, num_hidden_discriminator=1):
        super(Discriminator, self).__init__()
        self.dkp = dkp
        self.num_hidden_discriminator = num_hidden_discriminator
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_discriminator):
            self.hidden_layers.append(nn.Linear(d_hidden_size, d_hidden_size))
        
        self.output_layer = nn.Linear(d_hidden_size, num_labels + 1)
    
    def forward(self, x):
        layer_hidden = F.dropout(x, p=self.dkp, training=self.training)
        for layer in self.hidden_layers:
            layer_hidden = layer(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)
        
        flatten5 = layer_hidden
        
        logit = self.output_layer(layer_hidden)
        prob = F.softmax(logit, dim=1)
        
        return flatten5, logit, prob

# Example usage
# d_hidden_size = 128
# dkp = 0.5
# num_labels = 10
# num_hidden_discriminator = 1
# discriminator = Discriminator(d_hidden_size, dkp, num_labels, num_hidden_discriminator)
# x = torch.randn(32, d_hidden_size)  # Example input
# flatten5, logit, prob = discriminator(x)

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, class_dim, g_hidden_size, dkp, num_hidden_generator=1):
        super(ConditionalGenerator, self).__init__()
        self.dkp = dkp
        self.num_hidden_generator = num_hidden_generator

        # Define the input layer to combine z and class_dim
        self.input_layer = nn.Linear(z_dim + class_dim, g_hidden_size)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_generator):
            self.hidden_layers.append(nn.Linear(g_hidden_size, g_hidden_size))

        # Define the output layer
        self.output_layer = nn.Linear(g_hidden_size, g_hidden_size)

    def forward(self, z, class_labels):
        # Concatenate z and class_labels
        x = torch.cat((z, class_labels), dim=1)

        layer_hidden = self.input_layer(x)
        layer_hidden = F.leaky_relu(layer_hidden)
        layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)

        for layer in self.hidden_layers:
            layer_hidden = layer(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)

        layer_hidden = self.output_layer(layer_hidden)

        return layer_hidden

# Example usage
# z_dim = 100  # Dimension of the latent vector
# class_dim = 10  # Dimension of the class vector
# g_hidden_size = 128
# dkp = 0.5
# num_hidden_generator = 1
# generator = ConditionalGenerator(z_dim, class_dim, g_hidden_size, dkp, num_hidden_generator)
# z = torch.randn(32, z_dim)  # Example latent vector input
# class_labels = torch.randn(32, class_dim)  # Example class labels input
# generated_data = generator(z, class_labels)