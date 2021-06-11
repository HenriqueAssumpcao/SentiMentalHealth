import datetime
import pandas as pd

def findLast(my_list):
  for ix in range(len(my_list)-1,-1,-1):
    if my_list[ix]:
      return ix
  return None

def convert_utc(post):
    for dates in range(len(post.created_utc)):
        post.created_utc[dates] = datetime.datetime.fromtimestamp(post.created_utc[dates])
    return post

def check_missing_threads(post_df,comment_df):
    print("Checking for missing threads in comment_df...")
    for index,row in tqdm(post_df.iterrows()):
        for comment in row.comments:
            try:
                comment_df.loc[comment]
            except:
                try:
                    print(f"Comment:{comment} is missing from comment_df... Deleting thread")
                    post_df.drop(labels=index,inplace=True)
                except:
                    pass
