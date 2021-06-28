import nltk
import random
import torch
import time

import numpy as np
import pandas as pd

from datetime import timedelta
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, Sampler, SubsetRandomSampler
from .pytorchtools import EarlyStopping




SEED = 1234

def reset_seeds():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  # user defined functions
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch_idx, batch in enumerate(loader):
        if type(batch) is list:
          if len(batch) == 4:
            x1, len_x1, y, index = batch
          if len(batch) == 5:
            x1, len_x1, x2, y, index = batch
        else:
          x1, x2, len_x1, len_x2, y, index = (batch['X_b'],batch['X_t'],batch['b_len'],batch['t_len'],batch['y'],batch['ID'])

        optimizer.zero_grad()
        if getattr(model,'uses_two_series_as_input',False) == True:
          predictions, _ = model(x1.to(device), len_x1.to(device), x2.to(device), len_x2.to(device))
        else:
          if isinstance(model,nn.modules.transformer.Transformer):
            # sz = x1.size(1)
            # src_mask = _generate_square_subsequent_mask(sz)
            x1 = x1.permute(1,0,2)
            x2 = x2.unsqueeze(0)
            predictions = model(x1.to(device), x2.to(device))
            predictions = predictions[:,:,0].transpose(0,1)
          else:  
            predictions, _ = model(x1.to(device), len_x1.to(device))

        if isinstance(predictions,tuple):
            predictions = predictions[0]

        loss = criterion(predictions.squeeze(1), y.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion, device, return_predictions=False):    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    if return_predictions:
      list_predictions = []
    with torch.no_grad():
      for batch_idx, batch in enumerate(loader):
        if type(batch) is list:
          if len(batch) == 4:
            x1, len_x1, y, index = batch
          if len(batch) == 5:
            x1, len_x1, x2, y, index = batch
        else:
          x1, x2, len_x1, len_x2, y, index = (batch['X_b'],batch['X_t'],batch['b_len'],batch['t_len'],batch['y'],batch['ID'])

        if getattr(model,'uses_two_series_as_input',False) == True:
          predictions, _ = model(x1.to(device), len_x1.to(device), x2.to(device), len_x2.to(device))
        else:
          if isinstance(model,nn.modules.transformer.Transformer):
            # sz = x1.size(1)
            # src_mask = _generate_square_subsequent_mask(sz)
            x1 = x1.permute(1,0,2)
            x2 = x2.unsqueeze(0)
            predictions = model(x1.to(device), x2.to(device))
            predictions = predictions[:,:,0].transpose(0,1)
          else:  
            predictions, _ = model(x1.float().to(device), len_x1.to(device))

        #predictions = (predictions-.5)*2
        if return_predictions:
          list_predictions.append((predictions,y,index))

        if isinstance(predictions,tuple):
          predictions = predictions[0]

        loss = criterion(predictions.squeeze(1), y.to(device))
        epoch_loss += loss.item()

    if return_predictions:
      return epoch_loss / len(loader), list_predictions
    else:
      return epoch_loss / len(loader)

def train_over_nepochs(model, train_loader, valid_loader, criterion, device, patience=20, n_epochs=5, best_valid_loss=float('inf'),
                       filename=None, use_tune=False):
  if use_tune:
    from ray import tune

  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=2e-3)
  criterion = criterion.to(device)
  # initialize the early_stopping object
  epoch_time_list = []
  early_stopping = EarlyStopping(patience=patience, verbose=True)
  for epoch in range(n_epochs):
    train_iterator, valid_iterator = (iter(train_loader),iter(valid_loader))
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)
        
    end_time = time.time()
        
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    epoch_time_list.append(end_time-start_time)

    if use_tune:
      with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
      tune.report(train_loss=train_loss,val_loss=val_loss)
    else:
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          if filename is not None:
            torch.save(model.state_dict(), filename)

    if (epoch%2 == 0) or (epoch == n_epochs-1):    
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f}')
      print(f'\t Val. Loss: {valid_loss:.3f}')

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
      print("Early stopping")
      break

  # model.load_state_dict(torch.load('checkpoint.pt', map_location=lambda storage, loc: storage))

  return model, best_valid_loss, epoch_time_list

def createTensorDataset(src, src_len_series, y, max_length=None):

  src_len = torch.LongTensor(src_len_series)
  if max_length is not None:
    src_len[src_len > max_length] = max_length
  max_observed_len = src_len.max()

  return TensorDataset(src, src_len, y, torch.arange(len(src_len), dtype=torch.long))

def _subtreeHasValidBranches(grouped, root, branches, is_author=False, prev = []):
  prev = prev + [root]
  if not root in grouped.groups:
    if (is_author == True) and (len(prev)>2):
      branches.append(prev)
      return True
  else:
    comments       = grouped.get_group(root).comment_id.values
    is_post_author = grouped.get_group(root).is_post_author.values
    any_subtree_valid = np.any(
        [_subtreeHasValidBranches(grouped, comment, branches, is_author, prev) \
         for comment, is_author in zip(comments, is_post_author)]
        )
    if any_subtree_valid:
      return True
    else:
      if (is_author == True) and (len(prev)>2):
        branches.append(prev)
        return True

  return False

def getValidBranches(post):
  grouped = pd.DataFrame({'is_post_author':post.is_post_author[1:],'comment_id':post.comments,'parent_id':post.parent_id}).groupby('parent_id')
  # for name, group in grouped:
  #   print(group)
  branches = []
  if _subtreeHasValidBranches(grouped, post.name, branches):
    # map comments to indexes
    ind_branches = []
    # add 1 to indices since columns features, score and is_post_author start with post info 
    comment2ind = dict(zip(post.comments, range(1,len(post.comments)+1)))
    for bx, branch in enumerate(branches):
      # include post, i.e. index 0
      ind_branches.append([0]+[comment2ind[comment_id] for comment_id in branch[1:]])

    return ind_branches
  else:
    return None

def calculateParagraphScore(paragraph, sid, averaging=True):
  if averaging:
    sentence_list = nltk.tokenize.sent_tokenize(paragraph)
    return np.average([sid.polarity_scores(sentence)['compound'] for sentence in sentence_list])
  else:
    return sid.polarity_scores(paragraph)['compound']

def getNonOverlappingThreads(post_df):
  # create a groupby of authors containing their post/comment activities sorted by timestamp

  grouped = post_df.groupby('author', sort=False)
  post2seqlen = []
  for author, group in tqdm(grouped):

    # get all author activities in their own threads
    activities = []
    for index, row in group.iterrows():
      activities += [(row.created_utc[0],index,'self')]
      for jdx in range(1,row.seq_len):
        if row.is_post_author[jdx]:
          activities += [(row.created_utc[jdx], row.comments[jdx-1], index)]

    # create a dataframe
    activities.sort()
    activities_df = pd.DataFrame(activities)
    activities_df = activities_df.rename(columns={0:'created_utc', 1:'activity_id', 2:'link_id'})
    activities_df = activities_df.set_index('activity_id')

    # filter overlapping threads
    last_post = None
    last_comment = None
    for index, row in activities_df.iterrows():
      if row.link_id == 'self': # beginning of new thread sequence
        if (last_post is not None) & (last_comment is not None): # wrapping up old sequence
          post2seqlen.append((last_post, list(post_df.loc[last_post,'comments']).index(last_comment)+2))
        last_post = index
        last_comment = None
      else: # continuing one of the previous thread sequences
        if row.link_id == last_post: # continuing last thread
          last_comment = index
        else:
          if (last_post is not None) & (last_comment is not None): # finished last thread unexpectedly, wrap up and wait until next one begins
            post2seqlen.append((last_post, list(post_df.loc[last_post,'comments']).index(last_comment)+2))
          last_post = None
    if (last_post is not None) & (last_comment is not None): # wrapping up old sequence
      post2seqlen.append((last_post, list(post_df.loc[last_post,'comments']).index(last_comment)+2))
    
  tmp_series = pd.Series(dict(post2seqlen), name='num_comments', dtype=int)
  return tmp_series


# right now, we just use threads that have comments from author AND from others
def getThreadLen(post, comment_df):
  postid = post.name
  author_last_comment = -1
  has_other_commenters = False
  for ind in range(post.num_comments-1,-1,-1):
    comment_id = post.comments[ind]
    if author_last_comment == -1:
      if ind > 0 and comment_df.loc[comment_id].is_post_author:
        author_last_comment = ind
    elif not comment_df.loc[comment_id].is_post_author:
      has_other_commenters = True
    if author_last_comment != -1 and has_other_commenters:
      return author_last_comment+1
  return None

def areIntervalsShort(post, seqlen_series):
  postid = post.name
  if postid in seqlen_series.index:
    seqlen = seqlen_series.loc[postid]
    # from post (created_utc[0]) to last comment (created_utc[seqlen])
    timestamps = np.array(post.created_utc[:seqlen])
    timedeltas = timestamps[1:] - timestamps[:-1]
    return np.all(timedeltas < timedelta(days=1))#.total_seconds())


def extractFeatures(df, tokenizer, model, device, batch_size = 1024, max_paragraph_length=256):
  n = len(df)
  list_features = []
  for start in tqdm(range(0,n,batch_size)):
    batch = df[start:min(n,start+batch_size)]

    tokenized = batch['text'].apply((lambda x: tokenizer.encode(
        x[:max_paragraph_length], add_special_tokens=True, max_length=max_paragraph_length,
        truncation=True,
        )))
    padded = np.array([i + [0]*(max_paragraph_length-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.LongTensor(padded).to(device)
    attention_mask = torch.BoolTensor(attention_mask).to(device)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    list_features.append(last_hidden_states[0][:,0,:].cpu().numpy().astype(np.float16))

  features = pd.Series(np.vstack(list_features).tolist(), index=df.index, name='features')
  return features

  # Dataset

class RedditDataset(Dataset):
  """ Dataset representing reddit posts and comments """

  def __init__(self, post_df, col_name='score', max_thread_length=None, max_branch_length=None, thread_set=None):
    """
    Args:
        npz_file (string): Path to the npz file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    if thread_set is not None:
      thread_list = list(thread_set)
    else:
      thread_list = post_df.index

    self.post_df = post_df
    self.col_name = col_name
    self.max_thread_length = max_thread_length
    self.max_branch_length = max_branch_length
    self.thread_list = thread_list

    self.branches = [(name,bdx) \
                     for name, nbranches in post_df.loc[thread_list,'valid_branches'].apply(len).iteritems() \
                     for bdx in range(nbranches)]

  def __len__(self):
    return len(self.branches)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    max_thread_length = self.max_thread_length
    max_branch_length = self.max_branch_length
    post_df = self.post_df
    col_name = self.col_name
    branches = self.branches

    post_id, bdx = branches[idx]
    p = post_df.loc[post_id]
    b = p.valid_branches[bdx]
    y = p[col_name][b[-1]]

    # create tensor X_t (for thread): include all comments up to the last comment in b
    t_len = b[-1]    # thread len without author's last comment c is c's index
    t_len = min(max_thread_length,t_len)

    X_t = p.features[:t_len]
    if t_len < max_thread_length:
      X_t = F.pad(X_t, (0,0,0,max_thread_length-t_len))

    # create tensor X_b (for branch)
    b_sub = b[:min(max_branch_length,len(b)-1)]
    b_len = len(b_sub)

    X_b = p.features[b_sub]
    if b_len < max_branch_length:
      X_b = F.pad(X_b, (0,0,0,max_branch_length-b_len))


    sample = {'X_b': X_b, 'X_t': X_t, 'b_len':b_len, 't_len':t_len, 'y':y, 'ID': post_id}
    #sample = [X_b, X_t, b_len, t_len, y, post_id]
    return sample

def compute_error(results_df, pred_column_name, criteria, device):
  return {criterion.__class__.__name__.split('.')[-1]:
          float(criterion(
              torch.Tensor(results_df[pred_column_name].values).to(device),
              torch.Tensor(results_df['final score'].values).to(device))
          ) for criterion in criteria}