import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base_bert import BertPreTrainedModel
from bert import BertModel

from utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tokenizer import BertTokenizer
# from optimizer import AdamW

class Discriminator(nn.Module):
    def __init__(self, d_hidden_size, dkp, num_labels, num_hidden=1):
        super(Discriminator, self).__init__()
        self.dkp = dkp
        self.hidden_layers = nn.ModuleList([nn.Linear(d_hidden_size, d_hidden_size) for i in range(num_hidden)])        
        self.output_layer = nn.Linear(d_hidden_size, num_labels + 1)
    
    def forward(self, x):
        layer_hidden = F.dropout(x, p=self.dkp, training=self.training)
        for dense in self.hidden_layers:
            layer_hidden = dense(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)
        flatten = layer_hidden
        logit = self.output_layer(layer_hidden)
        prob = F.softmax(logit, dim=1)
        
        return flatten, logit, prob

# Example usage
# d_hidden_size = 128
# dkp = 0.5
# num_labels = 10
# num_hidden = 1
# discriminator = Discriminator(d_hidden_size, dkp, num_labels, num_hidden)
# x = torch.randn(32, d_hidden_size)  # Example input
# flatten5, logit, prob = discriminator(x)

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, class_dim, g_hidden_size, dkp, num_hidden=1):
        super(ConditionalGenerator, self).__init__()
        self.dkp = dkp

        # Conditional GAN: Define the input layer to combine z and class_dim
        self.input_layer = nn.Linear(z_dim + class_dim, g_hidden_size)

        self.hidden_layers = nn.ModuleList([nn.Linear(g_hidden_size, g_hidden_size) for i in range(num_hidden)])
        self.output_layer = nn.Linear(g_hidden_size, g_hidden_size)

    def forward(self, z, class_labels):
        # Conditional GAN: Concatenate z and class_labels
        x = torch.cat((z, class_labels), dim=1)
        layer_hidden = self.input_layer(x)
        layer_hidden = F.leaky_relu(layer_hidden)
        layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)
        for dense in self.hidden_layers:
            layer_hidden = dense(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dkp, training=self.training)
        layer_hidden = self.output_layer(layer_hidden)

        return layer_hidden

# Example usage
# z_dim = 100  # Dimension of the latent vector
# class_dim = 10  # Dimension of the class vector
# g_hidden_size = 128
# dkp = 0.5
# num_hidden = 1
# generator = ConditionalGenerator(z_dim, class_dim, g_hidden_size, dkp, num_hidden)
# z = torch.randn(32, z_dim)  # Example latent vector input
# class_labels = torch.randn(32, class_dim)  # Example class labels input
# generated_data = generator(z, class_labels)

class DiscriminatorBert(torch.nn.Module):
    def __init__(self, d_hidden_size, dkp, num_labels, num_hidden=1):
        super(DiscriminatorBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.discriminator = Discriminator(d_hidden_size, dkp, num_labels, num_hidden)

    def forward(self, input_ids, attention_mask):
        embedding = self.bert(input_ids, attention_mask)
        h_cls = embedding["pooler_output"]
        flatten, logit, prob = self.discriminator(h_cls)
        return flatten, logit, prob
    
def train():
    # From starter code
    # ---------------------------------------------------------------------------------------
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    # ---------------------------------------------------------------------------------------
    netG = ConditionalGenerator(768, num_labels, 128, 0.3, 1).to(device)
    netD = DiscriminatorBert(768, 0.3, num_labels, 1).to(device)
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            real_labels = data[1].to(device)  # Assuming class labels are provided with the data
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            # Concatenate real images and real labels
            real_labels_one_hot = torch.nn.functional.one_hot(real_labels, num_classes).float().to(device)
            real_combined = torch.cat((real_cpu, real_labels_one_hot), dim=1)
            
            # Forward pass real batch through D
            output = netD(real_combined).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, device=device)
            # Generate batch of class labels
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device)
            fake_labels_one_hot = torch.nn.functional.one_hot(fake_labels, num_classes).float().to(device)
            
            # Generate fake image batch with G
            fake = netG(noise, fake_labels_one_hot)
            label.fill_(fake_label)
            
            # Concatenate fake images and fake labels
            fake_combined = torch.cat((fake, fake_labels_one_hot), dim=1)
            
            # Classify all fake batch with D
            output = netD(fake_combined.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_combined).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1

if __name__ == "main":
    print("Starting Training Loop...")
    train()