import pickle
import numpy as np 
import torch 
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch.nn as nn 
from . import models
from tqdm import tqdm 
from .utils import reset_seeds, evaluate, train_over_nepochs

MAX_THREAD_LEN = 64 
def load_df(distilbert_filtered_posts_filename, max_length=MAX_THREAD_LEN):
  
  print(f'Loading {distilbert_filtered_posts_filename}...')
  if distilbert_filtered_posts_filename.endswith('.parquet'):
      post_df = pd.read_parquet(distilbert_filtered_posts_filename, columns=['features','score','is_post_author','seq_len','filtered_seqlen'])
  elif distilbert_filtered_posts_filename.endswith('.pkl'):
      post_df = pd.read_pickle(distilbert_filtered_posts_filename)
      post_df = post_df[['features','score','is_post_author','seq_len','filtered_seqlen']]

  post_df.features = post_df.apply(
      lambda p:torch.Tensor(np.hstack(
        (np.vstack(p.features[:min(max_length,p.seq_len-1)]).astype(np.float16),
         np.array(p.score[:min(max_length,p.seq_len-1)]).reshape(-1,1),
         np.array(p.is_post_author[:min(max_length,p.seq_len-1)]).reshape(-1,1))
      )), axis=1)
  
  return post_df

def compute_bin_weights(y, bin_width,min_value):
  return y.size(0)/(torch.bincount(((y - min_value)/bin_width).floor().int()) * 1./bin_width)

def save_stats_tensors(src_m, src_s, filepath):
  torch.save(src_m, f'{filepath}src_m.pt')
  torch.save(src_s, f'{filepath}src_s.pt')

def load_stats_tensors(filepath):
  src_m = torch.load(f'{filepath}src_m.pt')
  src_s = torch.load(f'{filepath}src_s.pt')
  return src_m, src_s

def get_znorm_params(post_df):
  src = np.concatenate(post_df.features.values)
  src_m = torch.Tensor(np.mean(src,axis=0, keepdims=True))
  src_s = torch.Tensor(np.std (src,axis=0, keepdims=True))
  return src_m, src_s

def split_indices(post_df):
  
    nthreads = len(post_df)     # number of threads
    if STRATIFIED:
        y = post_df.apply(lambda p: p.score[p.seq_len-1], axis=1).values
        bins = np.floor((y - MIN_VALUE)/BIN_WIDTH).astype(int)

        reset_seeds()
        fold_size = int(0.1*len(y))
        remaining_inds,valid_inds,_,_ = train_test_split(np.arange(nthreads),y,test_size=fold_size,stratify=bins)
        train_inds,test_inds,_,_ = train_test_split(remaining_inds,y[remaining_inds],test_size=fold_size,stratify=bins[remaining_inds])

        train_inds = np.sort(train_inds).tolist()
        valid_inds = np.sort(valid_inds).tolist()
        test_inds  = np.sort(test_inds).tolist()

    else:
        # divide randomly
        reset_seeds()
        assigned_set = np.random.multinomial(1,[.8,.1,.1],nthreads)
        
        train_inds = list(np.argwhere(assigned_set[:,0]).ravel())
        valid_inds = list(np.argwhere(assigned_set[:,1]).ravel())
        test_inds  = list(np.argwhere(assigned_set[:,2]).ravel())

    print(f"Number of training examples: {len(train_inds)}")
    print(f"Number of validation examples: {len(valid_inds)}")
    print(f"Number of testing examples: {len(test_inds)}")

    return train_inds, valid_inds, test_inds

def get_subreddit_weights(post_df, bin_width,min_value,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  y = torch.Tensor(post_df.apply(lambda p: p.score[p.seq_len-1], axis=1).values)
  subreddit2weights = {'all': compute_bin_weights(y,bin_width,min_value).to(device)}

  grouped = post_df.groupby(level=0)
  for name, group in grouped:
    y = torch.Tensor(group.apply(lambda p: p.score[p.seq_len-1], axis=1).values)
    subreddit2weights[name] = compute_bin_weights(y,bin_width,min_value).to(device)

  return subreddit2weights

def get_subreddit_range(post_df):
  subreddit2range = {}
  for subreddit in post_df.index.unique(level=0):
    subreddit2range[subreddit] = post_df.index.get_loc_level(subreddit)[0]
  subreddit2range['all'] = np.array([loc for v in subreddit2range.values() for loc in range(v.start,v.stop)])
  return subreddit2range  

# baseline "no changes"
def get_baselines_df(test_loader):

  batch = next(iter(test_loader))
  inds = batch[3].cpu().numpy()
  initial_score = pd.Series(batch[0][:,0,-2].cpu().numpy()*score_s+score_m,name='unchanged')
  final_score = pd.Series(batch[2].cpu().numpy(),name='final score')

  thread_mean = []   # baseline "average of comment scores in branch" 
  thread_last = []   # baseline "last comment scores in branch"
  thread_ncom = [] 
  for batch in test_loader:
      batch_size = batch[0].size(0)
      for ix in range(batch_size):
        ncom = int(batch[1][ix])
        thread_mean.append(float(batch[0][ix][:ncom ,-2].mean()*score_s+score_m))
        thread_last.append(float(batch[0][ix][ncom-1,-2])*score_s+score_m)
        thread_ncom.append(ncom)

  thread_mean = pd.Series(thread_mean,name='mean')
  thread_last = pd.Series(thread_last,name='last')
  thread_ncom = pd.Series(thread_ncom,name='thread ncom')

  df = pd.concat((initial_score,final_score,thread_ncom,thread_mean,thread_last),axis=1)
  # column_names=['unchanged','final score','thread ncom','mean','last']

  df = df.set_index(inds)
  return df

class WeightedL1Loss(torch.nn.Module):

  def __init__(self, bin_weights, bin_width,min_value):
    super(WeightedL1Loss,self).__init__()
    self.loss = nn.L1Loss(reduction='none')
    self.bin_weights = bin_weights
    self.bin_width = bin_width
    self.min_value = min_value
      
  def forward(self,inputs,targets):
    losses = self.loss(inputs,targets)
    weights = self.bin_weights[((targets - self.min_value)/self.bin_width).floor().long()]
    return (losses*weights).mean()

class WeightedMSELoss(torch.nn.Module):

  def __init__(self, bin_weights, bin_width,min_value):
    super(WeightedMSELoss,self).__init__()
    self.loss = nn.MSELoss(reduction='none')
    self.bin_weights = bin_weights
    self.bin_width = bin_width
    self.min_value = min_value

  def forward(self,inputs,targets):
    losses = self.loss(inputs,targets)
    weights = self.bin_weights[((targets - self.min_value)/self.bin_width).floor().long()]
    return (losses*weights).mean()

# TODO: delete code for plotting
def grid_search_train(train_loader,valid_loader,hidden_size_list, bidirectional_list, nlayers_dropout_list, train_criteria, test_criteria, results_filename,EMBEDDING_DIM,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), PLOT=False,N_EPOCHS=20,PATIENCE=3):
  dev_cpu = torch.device('cpu')

  # default params
  RNN_params = {
    'input_size': EMBEDDING_DIM,
    'output_size': 1,
    'uses_two_series_as_input': False,
    'dropout_out': 0.
  }
  vars = ['hidden_size', 'num_layers', 'dropout_rnn', 'bidirectional']

  train_criteria_name = [train_criterion.__class__.__name__.split('.')[-1] for train_criterion in train_criteria]
  test_criteria_name  = [ test_criterion.__class__.__name__.split('.')[-1] for test_criterion  in test_criteria ]

  try:
    # check if file exists
    results_df = pd.read_pickle(results_filename)
  except:
    # if not, create file
    results_df = pd.DataFrame(columns=['params','train_criterion'])
    results_df.train_criterion = results_df.train_criterion.astype(float)

  for zx, (train_criterion_name, train_criterion) in tqdm(enumerate(zip(train_criteria_name, train_criteria)), desc='L0: criterion'):
    for ix, hidden_size in tqdm(enumerate(hidden_size_list), desc='L1: hidden_size'):
      for jx, bidirectional in tqdm(enumerate(bidirectional_list), desc='L2: bidirectional'):
        for kx, (num_layers, dropout_rnn) in enumerate(nlayers_dropout_list):
          # set vars
          for v in vars:
            RNN_params[v] = eval(v)
          
          print(RNN_params)

          # see if row is present
          index = results_df.index[(results_df.params == RNN_params) & (results_df.train_criterion == train_criterion_name)]

          if len(index) == 0:
            valid_results = {'params':RNN_params.copy(), 'train_criterion':train_criterion_name}

            # instantiate model
            reset_seeds()
            model = models.GRUSentiment(RNN_params)
            
            # train until early stopping
            _, valid_loss, epoch_time_list = train_over_nepochs(
                model, train_loader, valid_loader, criterion=train_criterion,
                device=device, patience=PATIENCE, n_epochs=N_EPOCHS)
            
            # save parameters obtained at the end
            valid_results['para'] = copy.deepcopy(model.to(dev_cpu).state_dict())
            valid_results['epochs_time'] = epoch_time_list
            model.to(device)
            for test_criterion_name, test_criterion in zip(test_criteria_name, test_criteria):
                valid_results[test_criterion_name] = evaluate(model, iter(valid_loader),
                                                              criterion=test_criterion,
                                                              device=device)

            # reload parameters from best validation epoch    
            model.load_state_dict(torch.load('checkpoint.pt', map_location=lambda storage, loc: storage))

            # save parameters obtained at best validation epoch
            valid_results['para_best'] = copy.deepcopy(model.to(dev_cpu).state_dict())
            model.to(device)
            for test_criterion_name, test_criterion in zip(test_criteria_name, test_criteria):
              valid_results[test_criterion_name+'_best'] = evaluate(model, iter(valid_loader),
                                                            criterion=test_criterion,
                                                            device=device)         

            # save intermediate results        
            results_df = results_df.append(valid_results, ignore_index=True)
            results_df.to_pickle(results_filename)

          else:
            # get trained params
            valid_results = results_df.loc[index[0]]
            model = models.GRUSentiment(RNN_params)

            model.load_state_dict(valid_results.para)
            model = model.to(device)
            for test_criterion_name, test_criterion in zip(test_criteria_name, test_criteria):
              if np.isnan(valid_results[test_criterion_name]):
                results_df.at[index,test_criterion_name] = evaluate(model, iter(valid_loader),
                                                              criterion=test_criterion,
                                                              device=device)
            model.load_state_dict(valid_results.para_best)
            model = model.to(device)
            for test_criterion_name, test_criterion in zip(test_criteria_name, test_criteria):
              if np.isnan(valid_results[test_criterion_name]):
                results_df.at[index,test_criterion_name+'_best'] = evaluate(model, iter(valid_loader),
                                                              criterion=test_criterion,
                                                              device=device)

          # print summary
          best_valid_loss = results_df.WeightedL1Loss.min()
          print(f'XXXXX Best Val. Loss until now: {best_valid_loss:.3f}\n\n\n')
          
            
    #       break # for num_layers, dropout_rnn
    #     break # for bidirectional
    #   break # for hidden_size
    # break # for train_criterion_name


  return results_df