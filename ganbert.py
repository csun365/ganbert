import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base_bert import BertPreTrainedModel
from bert import BertModel
from classifier import *

from utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tokenizer import BertTokenizer
from tqdm import tqdm
import time
import math
import datetime
# from optimizer import AdamW

class Discriminator(nn.Module):
    def __init__(self, class_dim, input_dim=768, hidden_dim=256, num_hidden=1):
        super(Discriminator, self).__init__()
        self.dropout_rate = 0.2
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden)])        
        self.output_layer = nn.Linear(hidden_dim, class_dim + 1)
    
    def forward(self, x):
        layer_hidden = F.dropout(x, p=self.dropout_rate, training=self.training)
        layer_hidden = F.leaky_relu(self.input_layer(layer_hidden))
        layer_hidden = F.dropout(layer_hidden, p=self.dropout_rate, training=self.training)
        for dense in self.hidden_layers:
            layer_hidden = dense(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dropout_rate, training=self.training)
        flatten = layer_hidden
        logit = self.output_layer(layer_hidden)
        prob = F.softmax(logit, dim=1)
        return flatten, logit, prob

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, class_dim, output_dim=768, hidden_dim=256, num_hidden=1):
        super(ConditionalGenerator, self).__init__()
        self.dropout_rate = 0.2
        # Conditional GAN: Define the input layer to concatenate z and class_dim
        self.input_layer = nn.Linear(z_dim + class_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_hidden)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, class_labels):
        # Conditional GAN: Concatenate z and class_labels
        x = torch.cat((z, class_labels), dim=1)
        layer_hidden = self.input_layer(x)
        layer_hidden = F.leaky_relu(layer_hidden)
        layer_hidden = F.dropout(layer_hidden, p=self.dropout_rate, training=self.training)
        for dense in self.hidden_layers:
            layer_hidden = dense(layer_hidden)
            layer_hidden = F.leaky_relu(layer_hidden)
            layer_hidden = F.dropout(layer_hidden, p=self.dropout_rate, training=self.training)
        layer_hidden = self.output_layer(layer_hidden)
        return layer_hidden

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)

    args = parser.parse_args()
    return args

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(args):
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.device("mps")
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    train_dataset = SentimentDataset(train_data[:1000], args)
    dev_dataset = SentimentDataset(dev_data, args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    
    noise_size = 100
    epsilon = 1e-8 # for loss estimation
    # Models
    transformer = BertModel.from_pretrained('bert-base-uncased').to(device)
    generator = ConditionalGenerator(z_dim=noise_size, class_dim=num_labels).to(device)
    discriminator = Discriminator(class_dim=num_labels).to(device)

    # Inspired from Colab in original ganbert repo
    training_stats = []
    total_t0 = time.time()
    transformer_vars = [i for i in transformer.parameters()]
    d_vars = transformer_vars + [v for v in discriminator.parameters()]
    g_vars = [v for v in generator.parameters()]
    d_optimizer = optim.AdamW(d_vars, lr=5e-5)
    g_optimizer = optim.AdamW(g_vars, lr=5e-5)
    
    for epoch in range(10):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, 10))
        print('Training...')
        t0 = time.time()
        tr_g_loss = 0
        tr_d_loss = 0
        transformer.train()
        generator.train()
        discriminator.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            if step % 10 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # unpack data
            b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
            one_hot = torch.nn.functional.one_hot(b_labels, num_labels)
            real_batch_size = b_ids.shape[0]
            model_outputs = transformer(b_ids, attention_mask=b_mask)
            hidden_states = model_outputs["pooler_output"]
            noise = torch.zeros(real_batch_size, noise_size, device=device).uniform_(0, 1)
            gen_rep = generator(noise, one_hot)
            discriminator_input = torch.cat([hidden_states, gen_rep], dim=0)
            features, logits, probs = discriminator(discriminator_input)

            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]

            probs_list = torch.split(probs, real_batch_size)
            D_real_probs = probs_list[0]
            D_fake_probs = probs_list[1]

            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + epsilon))
            g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg

            # Disciminator's LOSS estimation
            logits = D_real_logits[:,0:-1]
            log_probs = F.log_softmax(logits, dim=-1)
            # The discriminator provides an output for labeled and unlabeled real data
            # so the loss evaluated for unlabeled data is ignored (masked)
            per_example_loss = -torch.sum(one_hot * log_probs, dim=-1)
            # per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
            labeled_example_count = per_example_loss.type(torch.float32).numel()

            # It may be the case that a batch does not contain labeled examples,
            # so the "supervised loss" in this case is not evaluated
            if labeled_example_count == 0:
                D_L_Supervised = 0
            else:
                D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)

            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
            d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U
            
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            d_loss.backward()
            g_optimizer.step()
            d_optimizer.step()
            tr_g_loss += g_loss.item()
            tr_d_loss += d_loss.item()

        avg_train_loss_g = tr_g_loss / len(train_dataloader)
        avg_train_loss_d = tr_d_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss generator: {0:.3f}".format(avg_train_loss_g))
        print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Test on Dev...")
        t0 = time.time()
        transformer.eval()
        discriminator.eval()
        generator.eval()
        total_test_accuracy = 0
        total_test_loss = 0
        nb_test_steps = 0
        all_preds = []
        all_labels_ids = []
        nll_loss = nn.CrossEntropyLoss(ignore_index=-1)
        for batch in dev_dataloader:
            b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
            with torch.no_grad():
                model_outputs = transformer(b_ids, attention_mask=b_mask)
                hidden_states = model_outputs["pooler_output"]
                _, logits, probs = discriminator(hidden_states)
                filtered_logits = logits[:,0:-1]
                total_test_loss += nll_loss(filtered_logits, b_labels)
            _, preds = torch.max(filtered_logits, 1)
            all_preds += preds.detach().cpu()
            all_labels_ids += b_labels.detach().cpu()

        all_preds = torch.stack(all_preds).numpy()
        print(np.stack((all_preds, all_labels_ids), axis=-1))
        all_labels_ids = torch.stack(all_labels_ids).numpy()
        test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
        avg_test_loss = total_test_loss / len(dev_dataloader)
        avg_test_loss = avg_test_loss.item()
        test_time = format_time(time.time() - t0)
        print("  Accuracy: {0:.3f}".format(test_accuracy))
        print("  Test Loss: {0:.3f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))
        training_stats.append({'epoch': epoch + 1,
                               'Training Loss generator': avg_train_loss_g,
                               'Training Loss discriminator': avg_train_loss_d,
                               'Valid. Loss': avg_test_loss,
                               'Valid. Accur.': test_accuracy,
                               'Training Time': training_time,
                               'Test Time': test_time})

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    config = SimpleNamespace(
        filepath='sst-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        dev_out = 'predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
        test_out = 'predictions/' + args.fine_tune_mode + '-sst-test-out.csv'
    )
    print("GANBERT!")
    train(config)