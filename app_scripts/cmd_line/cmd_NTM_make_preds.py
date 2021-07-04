print("Loading Libraries...Just a moment please!...")
import sys
import os
import umap
import hdbscan
from os import path
import torch
import pandas as pd
import joblib
import json
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer, util

def load_data(file_path):
    f = open(file_path, encoding="utf-8")
    list_ = []

    for line in f:
        list_.append(json.loads(line))
    corpus_train = pd.DataFrame(list_)
    return corpus_train


def related_articles(corpus):
    """
    Find the articles related to each content title and content
    """    
    name_regex = "[^]]+"
    url_regex = "http[s]?://[^)]+"
    markup_regex = '\[({0})]\(\s*({1})\s*\)'.format(name_regex, url_regex)    
    related_art = defaultdict(dict)
    for content, title in zip(corpus["content"], corpus["contentTitle"]):
        content = content.split("\n")
        for i, cont in enumerate(content):
            cont = cont.replace("*","")
            try:            
                for match in re.findall(markup_regex, cont):
                    related_art[title][match[0]] = match[1]
            except :
                print(cont)
    return related_art

def remove_hyperlinks_html_tags(corpus, col):
    """
    Remove hyperlinks from the content
    """
    fixed_content = []     
    for i, content in enumerate(corpus[col]):
        content = striphtml(content) 
        corpus[col][i] = content            
        content = content.split("\n")
        #print(content)
        for i, cont in enumerate(content): 
            if cont!='':
                if "https" in cont:
                    content[i] = ""
        fix_ = " ".join(content)
        fixed_content.append(fix_)
    corpus[col] = fixed_content
    return corpus

def striphtml(data):
    """
    Remove the html tags from the content
    """        
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def get_corpus(df, col):
    """
    Get related articles and clean art train data
    """
    rltd_articles = {}
    rltd_articles = related_articles(df)
    df = remove_hyperlinks_html_tags(df, col)
    return rltd_articles, df

class Sentence_Encoder:
    """ Sentence Embedder used for creating embedding models
    Arguments:
        embedding_model: The main embedding model to be used for extracting
                         document and embedding
    """
    def __init__(self,embedding_model=None):
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ValueError("Please select a correct SentenceTransformers model: msmarco-MiniLM-L-12-v3")


    def embed_documents(self, document, device, verbose = False):
        """ Embed a list of n words into an n-dimensional
        matrix of embeddings
        Arguments:
            document: A list of documents to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document embeddings with shape (n, m) with `n` documents
            that each have an embeddings size of `m`
        """
        embeddings = self.embedding_model.encode(document, device=device, show_progress_bar=verbose)
        return embeddings
    

def predict_pre_process(file_path, content, title):
    """
    Call functions and implement pre-processing for test dataset.
    """    
    if file_path:
        print("Loading file now...")
        art_train = load_data(file_path)      
        print("Pre-processing the dataset...")
        rltd_arts, art_train = get_corpus(art_train, content)
        print("Finalizing the dataset...")
        art_train["cont_title"] = art_train[title] + " " +art_train[content]
    return art_train, file_path, title

def sent_embed_dr_cl(documents, path, option, opath, upath, hpath, val):
    """
    Get sentence embeddings, umap embeddings, prediction on the test data together with the probabilities.
    """
    # Embedding model
    embedding_model = "msmarco-MiniLM-L-12-v3"
    device = "cuda" if torch.cuda.is_available else "cpu"
    new_embeddings = Sentence_Encoder(embedding_model).embed_documents(documents, device=device, verbose=True)
    
    if not os.path.exists(opath):
        raise ValueError("The required directory is not found. Please check if the directory exist and try again.")         
    if option == "sent_embeddings":
        print(f"Storing Sentence transformers embeddings at: {opath}new_articles_sent_embeddings.pt")        
        torch.save(new_embeddings, f"{opath}new_articles_sent_embeddings.pt") 

    elif option == "umap_embeddings":
        if not os.path.exists(upath):
            raise ValueError("The required UMAP directory is not found. Please check if the directory exist and try again.")                 
        if not os.path.exists(hpath):
            raise ValueError("The required HDBSCAN directory is not found. Please check if the directory exist and try again.")                                     
        umap_model = upath + val
        if not os.path.exists(umap_model):
            raise ValueError("The required UMAP file does not exist in the directory. Please check if the file exist and try again.")         
        else:
            print(f"Using UMAP saved model {val} saved at {upath}")        
            umap_model = joblib.load(f"{umap_model}")
            var = val.split("_")[-1]            
            hdb_model = hpath + f"hdbscan_model_{var}"
            print(f"Using HDBSCAN saved model hdbscan_model_{var} saved at {hpath}")                    
            hdbscan_model = joblib.load(f"{hdb_model}")            
            new_umap_embeddings = umap_model.transform(new_embeddings)
            predictions, probabilities = hdbscan.approximate_predict(hdbscan_model, new_umap_embeddings)
            topic_distr = hdbscan.membership_vector(hdbscan_model, new_umap_embeddings)    
            print(f"Storing UMAP embeddings at: {opath}new_articles_umap_embeddings.pt") 
            torch.save(new_embeddings, f"{opath}new_articles_umap_embeddings.pt")                                         

    if option=="umap_embeddings":
        return new_umap_embeddings, predictions, probabilities, val
    elif option=="umap_embeddings":
        return new_embeddings
     
    
def main(path, option, opath, content, title, upath=None, hpath=None , val=None):
    """
    Performs test data pre-processing, transforms the data to reduced dimensionality and makes predictions abotu the cluster labels
    """    
    art_train, file_path, title = predict_pre_process(path, content, title)
   
    content = list(art_train["cont_title"])
    titles = list(art_train[title])    

    if not os.path.exists(path):
        raise ValueError("The required input file is not found. Please check if the file exist and try again.")             
    if not os.path.exists(opath):
        raise ValueError("The required output path is not found. Please check if the output path exist and try again.")                                     
    if option == "sent_embeddings":
        sent_embeddings = sent_embed_dr_cl(content, path, option, opath, upath, hpath, val)    
    elif option == "umap_embeddings":
        umap_data, predictions, probabilities, model_number = sent_embed_dr_cl(content, path, option, opath, upath, hpath, val)    
        
        model_number = model_number.split("_")[-1].split(".")[0]
        prediction_df = pd.DataFrame(titles, columns=["Title"])
        prediction_df["predictions"] = predictions   
        prediction_df["probabilities"] = probabilities        
        prediction_df.to_csv(f"{opath}/predictions_file_{model_number}.csv")
    print("Thank you for using Neural Topic Modelling! :))")
    return

if __name__ == "__main__":    
    file = sys.argv[0]
    print(f"Starting to execute {file} script")  
    path = sys.argv[1]
    option = sys.argv[2]
    opath = sys.argv[3]   
    content = sys.argv[4]   
    title = sys.argv[5]   
    if option=="umap_embeddings":
        upath = sys.argv[6]                                       
        hpath = sys.argv[7]                                                                            
        val = sys.argv[8] 
        main(path, option, opath, content, title, upath, hpath, val)
    else:
        main(path, option, opath, content, title)
