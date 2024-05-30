'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from PCGRAD.pcgrad import PCGrad

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.dense_1 = nn.Linear(BERT_HIDDEN_SIZE, 256)
        self.dense_2 = nn.Linear(256, 128)
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
        self.conv1d_layer = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.SST_conv = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same')
        
        self.dense_sst = nn.Linear(BERT_HIDDEN_SIZE, 128)
        self.dense_para = nn.Linear(BERT_HIDDEN_SIZE, 128)
        self.dense_sts = nn.Linear(BERT_HIDDEN_SIZE, 128)
        
        self.output_sst = nn.Linear(2*128, N_SENTIMENT_CLASSES)
        
        self.para_1 = nn.Linear(4 * 128, 64)
        self.para_2 = nn.Linear(64, 1)
        self.sts_1 = nn.Linear(4 * 128, 64)
        self.sts_2 = nn.Linear(64, 1)
        

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        embedding = self.bert(input_ids, attention_mask)
        h_cls = embedding["pooler_output"]
        output = self.dropout_1(self.dense_2((self.dense_1(h_cls)))) #128
        
        conv = self.conv1d_layer(output.unsqueeze(2)) #128
        conv = conv.squeeze()
        #print(output.shape, conv.shape)
        return h_cls, conv # (8 x 128)


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # h_cls, shared_embed = self.forward(input_ids, attention_mask)
        # sst_embed = self.dense_sst(h_cls)
        # output = self.output_sst(torch.cat((shared_embed, sst_embed), dim=-1))
        # return F.relu(output)
        h_cls, conv = self.forward(input_ids, attention_mask)
        #print(h_cls.shape, conv.shape)
        
        sst_embed = self.dense_sst(h_cls) #128
        concat = torch.cat((conv, sst_embed), dim=-1)
        #print(concat.shape)
        concat = concat.unsqueeze(2)
        #print(concat.shape)
        
        self.SST_conv(concat)
        concat = concat.squeeze()
        
        return self.output_sst(concat)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        # h_cls_1, shared_embed_1 = self.forward(input_ids_1, attention_mask_1)
        # h_cls_2, shared_embed_2 = self.forward(input_ids_2, attention_mask_2)
        # para_embed_1 = self.dense_para(h_cls_1)
        # para_embed_2 = self.dense_para(h_cls_2)
        # output = self.output_para(torch.cat((shared_embed_1, para_embed_1, shared_embed_2, para_embed_2), dim=-1))
        # return output.squeeze(-1)
        
        
        h_cls_1, conv_1 = self.forward(input_ids_1, attention_mask_1)
        h_cls_2, conv_2 = self.forward(input_ids_2, attention_mask_2)
        
        para_embed_1 = self.dense_para(h_cls_1)
        para_embed_2 = self.dense_para(h_cls_2)
        
        output = self.para_1(torch.cat((para_embed_1, para_embed_2, conv_1, conv_2), dim=-1))
        output = self.para_2(F.relu(output))
        
        return output.squeeze(-1)
                
        embed_1 = self.forward(input_ids_1, attention_mask_1)
        embed_2 = self.forward(input_ids_2, attention_mask_2)
        output = self.para_1(torch.cat((embed_1, embed_2), dim=-1))
        return self.para_2(F.relu(output)).squeeze(-1)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        # h_cls_1, shared_embed_1 = self.forward(input_ids_1, attention_mask_1)
        # h_cls_2, shared_embed_2 = self.forward(input_ids_2, attention_mask_2)
        # sts_embed_1 = self.dense_para(h_cls_1)
        # sts_embed_2 = self.dense_para(h_cls_2)
        # output = self.output_para(torch.cat((shared_embed_1, sts_embed_1, shared_embed_2, sts_embed_2), dim=-1))
        # return output.squeeze(-1)
        
        h_cls_1, conv_1 = self.forward(input_ids_1, attention_mask_1)
        h_cls_2, conv_2 = self.forward(input_ids_2, attention_mask_2)
        
        sts_embed_1 = self.dense_sts(h_cls_1)
        sts_embed_2 = self.dense_sts(h_cls_2)
        
        output = torch.cat((sts_embed_1, sts_embed_2, conv_1, conv_2), dim = -1) #4 * 128 = 596
        output = self.sts_2(F.relu(self.sts_1(output))).squeeze(-1)
        return output
        
        embed_1 = self.forward(input_ids_1, attention_mask_1)
        embed_2 = self.forward(input_ids_2, attention_mask_2)
        output = self.sts_1(torch.cat((embed_1, embed_2), dim=-1))
        return self.sts_2(F.relu(output)).squeeze(-1)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.device("mps")
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data[:1000], args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data[:1000], args)
    
    para_train_data = SentencePairDataset(para_train_data[:1000], args)
    para_dev_data = SentencePairDataset(para_dev_data[:1000], args)
    
    sts_train_data = SentencePairDataset(sts_train_data[:1000], args)
    sts_dev_data = SentencePairDataset(sts_dev_data[:1000], args)
    

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    best_dev_acc = 0

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for sst_batch, para_batch, sts_batch in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            sst_b_ids, sst_b_mask, sst_b_labels = (sst_batch['token_ids'], sst_batch['attention_mask'], sst_batch['labels'])
            para_b_id1, para_b_mask1, para_b_id2, para_b_mask2, para_b_labels = (para_batch['token_ids_1'], para_batch['attention_mask_1'],
                                                                                para_batch['token_ids_2'], para_batch['attention_mask_2'],
                                                                                para_batch['labels'])
            sts_b_id1, sts_b_mask1, sts_b_id2, sts_b_mask2, sts_b_labels = (sts_batch['token_ids_1'], sts_batch['attention_mask_1'],
                                                                                sts_batch['token_ids_2'], sts_batch['attention_mask_2'],
                                                                                sts_batch['labels'])

            sst_b_ids = sst_b_ids.to(device, dtype = torch.long)
            sst_b_mask = sst_b_mask.to(device, dtype = torch.long)
            sst_b_labels = sst_b_labels.to(device, dtype = torch.long)
            
            para_b_id1 = para_b_id1.to(device, dtype = torch.long)
            para_b_mask1 = para_b_mask1.to(device, dtype = torch.long)
            para_b_id2 = para_b_id2.to(device, dtype = torch.long)
            para_b_mask2 = para_b_mask2.to(device, dtype = torch.long)
            para_b_labels = para_b_labels.to(device, dtype = torch.long)
            
            sts_b_id1 = sts_b_id1.to(device, dtype = torch.long)
            sts_b_mask1 = sts_b_mask1.to(device, dtype = torch.long)
            sts_b_id2 = sts_b_id2.to(device, dtype = torch.long)
            sts_b_mask2 = sts_b_mask2.to(device, dtype = torch.long)
            sts_b_labels = sts_b_labels.to(device, dtype = torch.long)

            optimizer.zero_grad()
            sst_logits = model.predict_sentiment(sst_b_ids, sst_b_mask) 
            para_logits = model.predict_paraphrase(para_b_id1, para_b_mask1, para_b_id2, para_b_mask2)
            sts_logits =  model.predict_similarity(sts_b_id1, sts_b_mask1, sts_b_id2, sts_b_mask2)
            
            sst_loss = F.cross_entropy(sst_logits, sst_b_labels.view(-1), reduction='sum') / args.batch_size
            para_loss = F.cross_entropy(para_logits, para_b_labels.float(), reduction='sum') / args.batch_size
            sts_loss = F.mse_loss(sts_logits, sts_b_labels.float(), reduction='sum') / args.batch_size
            
            loss = sst_loss + para_loss + sts_loss
            losses = [sst_loss, para_loss, sts_loss]

            optimizer.pc_backward(losses)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_sst_acc, _ , _ , train_para_acc, _ , _ , train_sts_corr, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_sst_acc, _ , _ , dev_para_acc, _ , _ , dev_sts_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_acc = (train_sst_acc + train_para_acc + train_sts_corr)/3
        dev_acc = (dev_sst_acc + dev_para_acc + dev_sts_corr)/3
        
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        
    save_model(model, optimizer, args, config, args.filepath)

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        device = torch.device("mps")
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data[:1000], shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data[:1000], shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data[:1000], args)
        para_dev_data = SentencePairDataset(para_dev_data[:1000], args)

        para_test_dataloader = DataLoader(para_test_data[:1000], shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data[:1000], shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)