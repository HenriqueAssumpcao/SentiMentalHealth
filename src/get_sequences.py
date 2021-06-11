#Script pra pegar a sequência das árvores - autor do post

#set index
data = data.set_index('id')
data["link_id"] = data[data.tipo == "comments"]["link_id"].apply(lambda x: x[3:])
#remvove excexx space
data["text"] = data["text"].map(str.strip)
#sort
data = data.sort_values(by="created_utc")
#remove posts que tem autores [deleted]
data[data["tipo"] == "posts"] = data[data["tipo"] == "posts"][data.author != "[deleted]"]
#remove tudo que é [deleted]
data = data[data["text"] != "[deleted]"]
#juntar title e text dos posts
#data["text"] = data[(data["tipo"]=="posts")]["title"]+ " \n "+data[(data["tipo"]=="posts")]["text"]
data.loc[(data["tipo"]=="posts"), "text"] = data[(data["tipo"]=="posts")]["title"]+ " \n "+data[(data["tipo"]=="posts")]["text"]
#calcula count_words
data["count_words"] = data.loc[:,"text"].apply(lambda x : len([re.sub('[^0-9a-zA-Z]+', '', w) for w in str(x).split(" ")]))
#pego tudo que é maior que 3
data = data[data["count_words"] >= 3]
#Pego apenas os posts que tem comentarios
id_posts = list(data[data["tipo"] == "posts"].index)
id_posts_with_linkid = list(set(data[data["link_id"].isin(id_posts)]["link_id"].values))
#Pegar sequencias de atividades
print("Qtde. threads a serem analisadas = "+str(len(id_posts_with_linkid)))
sequences_activities = {}

k = 0

for id_ in id_posts_with_linkid:
    if(k%1000==0):
        print(subreddit + " " +str(k)+"/"+str(len(id_posts_with_linkid)))
    author_posts, data_post = data.loc[id_][["author", "created_utc"]].values
    thread = data[data["link_id"] == id_]
    sequences_activities[id_] = []
    comms_inter = 0
    for index, row in thread.iterrows():
        if(row["author"] != author_posts):
            comms_inter += 1
        elif(comms_inter > 0): #pra garantir que teve uma interação depois do post...
            #encontrei uma sequencia
            #check se o usuário teve atividade em outra thread:
            data_comm = row["created_utc"]
            if(len(data[(data["author"] == row["author"]) & ((data["created_utc"] > data_post) & (data["created_utc"] < data_comm))]) == 0):
                sequences_activities[id_].append(index)
                #reset
                ## data_post é a data mais antiga
                data_post = data_comm
            ## tem que ter interações antes...
            comms_inter = 0  
    k+=1
pickle.dump(sequences_activities, open( "results/sequences_activities_"+subreddit+"_corteTamnho3.p", "wb" ))

#Script pra pegar toda a sequência das árvores - autor do post + outros usuários

sequences_activities_all_thread = {}
k = 0
for id_post in list(sequences_activities.keys()):
    if(k%1==0):
        print(subreddit + " " +str(k)+"/"+str(len(sequences_activities.keys())))
    sequences_activities_all_thread[id_post] = []
    comm1 = id_post
    for i in range(0,len(sequences_activities[id_post])):
        comm2 = sequences_activities[id_post][i]
        thread = data[data["link_id"] == id_post]

        if(i == 0): #inclui o post
            comms_inter = list(thread[(thread["created_utc"] > data.loc[comm1]["created_utc"]) & (thread["created_utc"] < thread.loc[comm2]["created_utc"])].index)
        else:
            comms_inter = list(thread[(thread["created_utc"] > thread.loc[comm1]["created_utc"]) & (thread["created_utc"] < thread.loc[comm2]["created_utc"])].index)
        comm1 = comm2
        sequences_activities_all_thread[id_post].extend(comms_inter)
        sequences_activities_all_thread[id_post].extend([comm2])
    k+=1
	
pickle.dump(sequences_activities_all_thread, open("results/sequences_activities_all_thread_"+subreddit+"_corte_corteTamnho3.p", "wb" ))