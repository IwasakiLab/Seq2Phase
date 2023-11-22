import argparse

def main(cpu, thread, less_data, client, scaffold, nonllps):
    from sklearn.svm import SVC
    from Bio import SeqIO
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import random
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from joblib import dump
    from bio_embeddings.embed.prottrans_t5_embedder import ProtTransT5XLU50Embedder
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
    from nn_model import NN
    
    if thread != -1:
        torch.set_num_threads(thread)
        
    if client:
        client_dict = dict(SeqIO.index(client, "fasta"))
    else:
        client_dict = dict(SeqIO.index("data/drllps_client_clstr_Homo_sapiens.fasta", "fasta"))
    if nonllps:
        nonllps_dict = dict(SeqIO.index(nonllps, "fasta"))
    else:
        nonllps_dict = dict(SeqIO.index("data/drllps_nonllps_clstr_Homo_sapiens.fasta", "fasta"))
    if scaffold:
        scaffold_dict = dict(SeqIO.index(scaffold, "fasta"))
    else:
        scaffold_dict = dict(SeqIO.index("data/drllps_scaffold_clstr_Homo_sapiens.fasta", "fasta"))

    if not less_data:
        list_client=[str(client_dict[k].seq) for k in client_dict if len(client_dict[k].seq)<=10000]
        list_nonllps=[str(nonllps_dict[k].seq) for k in nonllps_dict if len(nonllps_dict[k].seq)<=10000]
        list_scaffold=[str(scaffold_dict[k].seq) for k in scaffold_dict if len(scaffold_dict[k].seq)<=10000]
    else:
        list_client=[str(client_dict[k].seq) for k in client_dict if len(client_dict[k].seq)<=1000]
        list_nonllps=[str(nonllps_dict[k].seq) for k in nonllps_dict if len(nonllps_dict[k].seq)<=1000]
        list_scaffold=[str(scaffold_dict[k].seq) for k in scaffold_dict if len(scaffold_dict[k].seq)<=1000]

    print("Loading embedder model...")
    if cpu or not torch.cuda.is_available():
        embedder = ProtTransT5XLU50Embedder(device='cpu')
    else:
        embedder = ProtTransT5XLU50Embedder(device='cuda')
    
    print("Embedding...")
    
    if not less_data:
        x_client = embedder.embed_many(list_client)
        x_client = list(x_client)
        x_client = np.array([i.mean(axis=0) for i in x_client])
        
        x_scaffold = embedder.embed_many(list_scaffold)
        x_scaffold = list(x_scaffold)
        x_scaffold = np.array([i.mean(axis=0) for i in x_scaffold])
        
        x_nonllps = embedder.embed_many(list_nonllps)
        x_nonllps = list(x_nonllps)
        x_nonllps = np.array([i.mean(axis=0) for i in x_nonllps])
    else:
        random.seed(0)
        x_client = embedder.embed_many(list_client)
        x_client = list(x_client)
        x_client = np.array([i.mean(axis=0) for i in x_client])
        
        x_scaffold = embedder.embed_many(list_scaffold)
        x_scaffold = list(x_scaffold)
        x_scaffold = np.array([i.mean(axis=0) for i in x_scaffold])
        
        try:
            x_nonllps = embedder.embed_many(random.sample(list_nonllps, len(list_client)))
        except ValueError:
            x_nonllps = embedder.embed_many(list_nonllps)
        x_nonllps = list(x_nonllps)
        x_nonllps = np.array([i.mean(axis=0) for i in x_nonllps])
    
    print("Training client model...")
    if not less_data:
        x=np.concatenate([x_client,x_nonllps], axis=0)
        y=np.array([True]*len(x_client) + [False]*len(x_nonllps))
    else:
        try:
            x = np.concatenate([x_client, x_nonllps[np.random.choice(len(x_nonllps), len(x_client), replace=False), :]], axis=0)
            y=np.array([True]*len(x_client) + [False]*len(x_client))
        except ValueError:
            x=np.concatenate([x_client,x_nonllps], axis=0)
            y=np.array([True]*len(x_client) + [False]*len(x_nonllps))
    
    estimators = [
        ('nn', NN(lr=0.01)),
        ('rf', RandomForestClassifier(max_depth=20, max_features="sqrt", class_weight="balanced",n_estimators=200, n_jobs=thread)),
        ('svm', make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True, gamma="auto"))),
        ('hgboost', HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=80))
    ]
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    model_client=StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(class_weight="balanced"), n_jobs=thread, cv=cv
    )
        
    model_client.fit(x,y)
    dump(model_client, 'trained_model_client.joblib')
    
    print("Training scaffold model...")
    if not less_data:
        x = np.concatenate([x_scaffold, x_client, x_nonllps], axis=0)
        y = np.array([True]*len(x_scaffold) + [False]*(len(x_client)+len(x_nonllps)))
    else:
        np.random.seed(0)
        try:
            x = np.concatenate([x_scaffold, x_client[np.random.choice(len(x_client), len(x_scaffold), replace=False),:], x_nonllps[np.random.choice(len(x_nonllps), len(x_scaffold), replace=False),:]], axis=0)
            y = np.array([True]*len(x_scaffold) + [False]*(len(x_scaffold)*2))
        except ValueError:
            x = np.concatenate([x_scaffold, x_client, x_nonllps], axis=0)
            y = np.array([True]*len(x_scaffold) + [False]*(len(x_client)+len(x_nonllps)))
        
    estimators = [
        ('nn', make_pipeline(StandardScaler(), PCA(n_components=128), NN(lr=0.05))),
        ('rf', make_pipeline(StandardScaler(), PCA(n_components=128), RandomForestClassifier(max_depth=5, max_features="log2", class_weight="balanced", n_estimators=200, n_jobs=thread))),
        ('svm', make_pipeline(StandardScaler(), PCA(n_components=128), SVC(class_weight="balanced", probability=True, C=1, kernel="rbf", gamma="scale"))),
        ('hgboost', make_pipeline(StandardScaler(), PCA(n_components=64), HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40)))
    ]
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    model_scaffold=StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(class_weight="balanced"), n_jobs=thread, cv=cv
    )
        
    model_scaffold.fit(x,y)
    dump(model_scaffold, 'trained_model_scaffold.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Seq2Phase model')

    parser.add_argument('--cpu', action='store_true', help='Use CPU for embedding (default: False)')
    parser.add_argument('-t', '--thread', type=int, default=-1, help='number of threads on CPU. -1 for all threads (default: -1)')
    parser.add_argument('--less_data', action='store_true', help='Process with less data. Recommended when using CPU. (default: False)')
    parser.add_argument('--client', default=None, help='Path to your own client fasta file, if you want to override the default one.')
    parser.add_argument('--scaffold', default=None, help='Path to your own scaffold fasta file, if you want to override the default one.')
    parser.add_argument('--nonllps', default=None, help='Path to your own Non-LLPS fasta file, if you want to override the default one.')

    args = parser.parse_args()

    main(args.cpu, args.thread, args.less_data, args.client, args.scaffold, args.nonllps)