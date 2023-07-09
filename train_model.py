from sklearn.svm import SVC
from Bio import SeqIO
import numpy as np
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump
from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder

sequences = []
for record in SeqIO.parse("data/swiss_prot_human_220916.fasta", "fasta"):
    sequences.append(record)
seq_dict={str(rec.id):str(rec.seq) for rec in sequences}

client_set=set()
scaffold_set=set()
others_set=set()
for rec in SeqIO.parse("data/drllps_client_clstr_Homo_sapiens.fasta", "fasta"):
    client_set.add(rec.id)
for rec in SeqIO.parse("data/drllps_nonllps_clstr_Homo_sapiens.fasta", "fasta"):
    others_set.add(rec.id)
for rec in SeqIO.parse("data/drllps_scaffold_clstr_Homo_sapiens.fasta", "fasta"):
    scaffold_set.add(rec.id)

list_client=[]
list_others=[]
list_scaffold=[]
for k in seq_dict.keys():
    if k in others_set:
        list_others.append(seq_dict[k])
    elif k in client_set:
        list_client.append(seq_dict[k])
    elif k in scaffold_set:
        list_scaffold.append(seq_dict[k])
        
def under_sampling(x, y):
    x_ture=x[y==True]
    x_false=x[y==False]
    y_ture=y[y==True]
    y_false=y[y==False]
    positive_n=len(y_ture)
    negative_n=len(y_false)
    random_index=np.random.randint(0,negative_n,positive_n)  
    x_false_u=x_false[random_index]
    y_false_u=y_false[random_index]
    return np.concatenate([x_ture, x_false_u]), np.concatenate([y_ture, y_false_u])

np.random.seed(0)
x_all=np.array(list_client+list_others)
y_all=np.array([True]*len(list_client) + [False]*len(list_others))
x,y=under_sampling(x_all,y_all)

print("Loading embedder model...")
embedder = ProtTransT5XLU50Embedder(device='cuda:0')

print("Embedding for the client model...")
x = embedder.embed_many(x)
x=list(x)
x=[i.mean(axis=0) for i in x]

print("Training the client model...")
model_client=make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True))
model_client.fit(x,y)
dump(model_client, 'trained_model_client.joblib')

x_all=np.array(list_scaffold+list_others)
y_all=np.array([True]*len(list_scaffold) + [False]*len(list_others))
x,y=under_sampling(x_all,y_all)

print("Embedding for the scaffold model...")
x = embedder.embed_many(x)
x=list(x)
x=[i.mean(axis=0) for i in x]

print("Training the scaffold model...")
model_scaffold=make_pipeline(StandardScaler(), 
                    PCA(n_components=32),
                    StandardScaler(),
                    SVC(class_weight="balanced", probability=True))
model_scaffold.fit(x,y)
dump(model_scaffold, 'trained_model_scaffold.joblib')