import requests
import datetime
from timeit import default_timer
import pickle
import itertools
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import time
from ratelimit import limits, sleep_and_retry
import os
import argparse 

"""First we will get all reddit submissions to a given subreddit, in each day. Since there is a limit of 500 results per request, let's hope the number of submissions is smaller than that. All we need is the submission id and content."""


parser = argparse.ArgumentParser()
parser.add_argument("--subreddit",type=str,default="SuicideWatch",help="subreddit to scrape")
opt = parser.parse_args()
datadir = os.getcwd() + "/data/"


dt_curr = int(datetime.datetime(2017,1,1,0,0).timestamp())
dt_last = int(datetime.datetime(2017, 1,2,0,0).timestamp())
post_endpoint=' https://api.pushshift.io/reddit/search/submission/'
post_parameters = {
    'after': dt_curr,
    'before': dt_last,
    'fields': ('author','selftext','id','created_utc','num_comments'),
    'size': 500,
    'sort': 'asc'
}

print('Collecting posts...')
subreddit = opt.subreddit

post_parameters['subreddit'] = subreddit
post2data = dict()
nresults = 100

@sleep_and_retry
@limits(calls=6, period=4)
def fetch(session, post_parameters):
    for attempts in range(100):
        response = session.get(post_endpoint,params=post_parameters)
        if response.status_code != 200:
            print('Error at timestamp {}. Retrying...'.format(post_parameters['after']))
        else:
            return response

outname=datadir + subreddit+'_post2data.pkl'
if os.path.exists(outname):
    with open(outname,'rb') as infile:
       post2data = pickle.load(infile)
else:
  while nresults == 100:
      print('Getting posts created after', datetime.datetime.fromtimestamp(post_parameters['after']))
      with requests.Session() as session:
          response = fetch(session, post_parameters)
          nresults = len(response.json()['data'])
          for post in response.json()['data']:
              post_id = post['id']
              del post['id']
              post2data[post_id]=post
          post_parameters['after'] = post['created_utc']

  with open(outname,'wb') as outfile:
      pickle.dump(post2data,outfile)

"""Now we will collect the ids of the comments associated with each post."""

import time
base_url = 'https://api.pushshift.io/reddit/submission/comment_ids/'
post_ids = [post_id for post_id, post_data in post2data.items() if post_data['num_comments']>0]
nposts = len(post_ids)
post2comments = dict()

print('Collecting comment ids...')
start_time = default_timer()

@sleep_and_retry
@limits(calls=6, period=6)
def fetch(session, i):
    post_id = post_ids[i]
    if i%10 == 0:
        print('(Elapsed {}s) Processing post # {} of {}'.format(int(default_timer()-start_time),i,nposts) )
    for attempts in range(100):
        response = session.get(base_url+post_id)
        if response.status_code == 200:
            break
        else:
            print('Error (too many requests). Retrying after some random amount of time.')
            time.sleep(random.random())
            
    return post_id,response

async def get_data_asynchronous():
    with ThreadPoolExecutor(max_workers=6) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `fetch`

            # Initialize the event loop        
            loop = asyncio.get_event_loop()
            
            # Use list comprehension to create a list of
            # tasks to complete. The executor will run the `fetch`
            # function for each csv in the csvs_to_fetch list
            tasks = [
                loop.run_in_executor(
                    executor,
                    fetch,
                    *(session, ind) # Allows us to pass in multiple arguments to `fetch`
                )
                for ind in range(nposts)
            ]
            
            # Initializes the tasks to run and awaits their results
            for post_id,response in await asyncio.gather(*tasks):
                if response.status_code != 200:
                    print('Error at url {}'.format(response.url))
                else:
                    post2comments[post_id] = response.json()['data']

outname=datadir + subreddit +'_post2comments.pkl'
if os.path.exists(outname):
  with open(outname,'rb') as infile:
    post2comments = pickle.load(infile)
else:                        
  loop = asyncio.get_event_loop()
  future = asyncio.ensure_future(get_data_asynchronous())
  loop.run_until_complete(future)
  with open(outname,'wb') as outfile:
      pickle.dump(post2comments,outfile)

comment_ids = list(itertools.chain.from_iterable(post2comments.values()))
print(len(comment_ids))
print(len(post2comments))

"""Now we will collect the comments."""

comment_endpoint='https://api.pushshift.io/reddit/comment/search'
# comment_parameters = {
#     'fields': ('author','body','link_id','parent_id','id','created_utc'),
#     'sort': 'asc'
# }

print('Collecting comments...')

base_url = comment_endpoint+'?sort=asc&fields=author,body,link_id,parent_id,id,created_utc&ids='
comment2data = dict()
ncomments = len(comment_ids)
start_time = default_timer()

@sleep_and_retry
@limits(calls=6, period=6)
def fetch(session, i):
    full_url = base_url+','.join(comment_ids[i:min(i+1000,ncomments)])
    if i%10000 == 0:
        print('(Elapsed {}s) Processing comment # {} of {}'.format(int(default_timer()-start_time),i,ncomments) )
    for attempts in range(100):
        response = session.get(full_url)
        if response.status_code == 200:
            break
        else:
            print('Error (too many requests). Retrying after some random amount of time.')
            time.sleep(random.random())

    return response

async def get_data_asynchronous():
    with ThreadPoolExecutor(max_workers=6) as executor:
        with requests.Session() as session:
            # Set any session parameters here before calling `fetch`

            # Initialize the event loop        
            loop = asyncio.get_event_loop()
            
            
            # Use list comprehension to create a list of
            # tasks to complete. The executor will run the `fetch`
            # function for each csv in the csvs_to_fetch list
            tasks = [
                loop.run_in_executor(
                    executor,
                    fetch,
                    *(session, i) # Allows us to pass in multiple arguments to `fetch`
                )
                for i in range(0,ncomments,1000)
            ]
            
            # Initializes the tasks to run and awaits their results
            for response in await asyncio.gather(*tasks):
                if response.status_code != 200:
                    print('Error at {}-th comment_id ()'.format(i,comment_ids[i]))
                else:
                    for comment in response.json()['data']:
                        comment_id = comment['id']
                        del comment['id']
                        comment2data[comment_id]=comment

outname=datadir + subreddit+'_comment2data.pkl'
if os.path.exists(outname):
  with open(outname,'rb') as infile:
    comment2data = pickle.load(infile)
else:                            
  loop = asyncio.get_event_loop()
  future = asyncio.ensure_future(get_data_asynchronous())
  loop.run_until_complete(future)
  with open(outname,'wb') as outfile:
    pickle.dump(comment2data,outfile)
