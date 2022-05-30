import os
import sys
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from transformers import AutoTokenizer
from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)





class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,fp16):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(2*input_size, hidden_size, num_layers, batch_first=True,dropout=0.20,bidirectional=True)
        self.fp16=fp16
        #now we insert 2* input size because each element of the batch will be inserted to the gru
        #with double representation - explained below
        self.fc = nn.Linear(2*hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        if self.fp16:
            h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size,dtype=torch.float16).to('cuda') 
        #since we have a biderictional gru, hidden  should have the double dimension of the gru layers as the first dimension
        # Morevoer we will use half precision floats since we implement fp 16 optimization
        else:
            h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to('cuda') 
        # Forward propagate gru
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8,fp16=True):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        
        input_size = self.bert.config.hidden_size
        #gru layer
        self.fc=RNN(input_size,128,1,2,fp16) 
        self.tokenizer=get_tokenizer(lm)
        self.sep_token_id=self.tokenizer.sep_token_id


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary
             prediction
        """
        def mapping(x):
            map_x=torch.torch.zeros_like(x).long()
            map_x = map_x.to(self.device)
            map_x=torch.where(x==self.sep_token_id,1,map_x) #(batch_size,emb_size)
            map_x=torch.reshape(map_x,(map_x.shape[0],1,map_x.shape[1])) #(batch_size,1,emb_size)
            if self.fp16:
                return map_x.type(torch.cuda.HalfTensor)#We have to modify map_x tensr's type to halftensor (float16), since 
                #ditto will be trained with fp16 optimization module on
            else:
                return map_x
        
        batch_size = len(x1)
        x1 = x1.to(self.device) # (batch_size, seq_len)
        map_x1=mapping(x1)

        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            map_x2=mapping(x2)
            
            enc = self.bert(torch.cat((x1, x2)))[0] #(2*batch_size,seq_len,emb_size)
            
           
            enc1_new = enc[:batch_size] # (batch_size,seq_len, emb_size)
            enc2_new = enc[batch_size:] # (batch_size, seq_len, emb_size)
            enc1_new=torch.matmul(map_x1,enc1_new) # (batch_size,1, emb_size)
            enc2_new=torch.matmul(map_x2,enc2_new) # (batch_size,1, emb_size)


            enc_old =enc[:, 0, :]
            batch_size = len(x1)
            enc1_old = enc_old[:batch_size] # (batch_size, emb_size)
            enc2_old = enc_old[batch_size:] # (batch_size, emb_size)
            enc1_old=torch.reshape(enc1_old,(enc1_old.shape[0],1,enc1_old.shape[1]))
            enc2_old=torch.reshape(enc2_old,(enc2_old.shape[0],1,enc2_old.shape[1]))
            enc1=torch.cat((enc1_old,enc1_new),dim=2)# (batch_size,1, 2*emb_size) 
            enc2=torch.cat((enc2_old,enc2_new),dim=2)# (batch_size,1, 2*emb_size)
            #each of the 2 above tensores has the info of the CLS token in the first emb_size and
            # the info from the SEP token in the second part of the emb_size. So we serialize it
            # and we have 2*emb_size

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam) # (batch_size,1, 2*emb_size) 
    
        else:
            enc = self.bert(x1)[0] # (batch_size,seq_len, emb_size)
            enc_new=torch.matmul(map_x1,enc) # (batch_size,1, emb_size)
            enc = enc[:,0,:] # (batch_size, emb_size)
            enc = torch.reshape(enc,(enc.shape[0],1,enc.shape[1])) # (batch_size,1, emb_size)
            enc=torch.cat((enc,enc_new),dim=2) # (batch_size,1, 2*emb_size)
        
        

        return self.fc(enc) # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)

        loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug,fp16=hp.fp16)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()
