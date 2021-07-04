from pre_process import *

def sent_embed_dr_cl(documents, path):
    """
    Get sentence embeddings, umap embeddings, prediction on the test data together with the probabilities.
    """
    # Embedding model
    embedding_model = "msmarco-MiniLM-L-12-v3"
    new_embeddings = Sentence_Encoder(embedding_model).embed_documents(documents, verbose=True)    

    hdbscan_model = joblib.load(f"{path}/hdbscan_model.sav")
    umap_model = joblib.load(f"{path}/umap_model.sav") 
    

    new_umap_embeddings = umap_model.transform(new_embeddings.to(device))
    predictions, _ = hdbscan.approximate_predict(hdbscan_model, new_umap_embeddings)
    probabilities = hdbscan.membership_vector(hdbscan_model, new_umap_embeddings)     
    
    return new_embeddings, new_umap_embeddings, predictions, probabilities    
     
    
def main():
    """
    Performs test data pre-processing, transforms the data to reduced dimensionality and makes predictions abotu the cluster labels
    """
    art_train, file_path, title = predict_pre_process()
   
    content = list(art_train["cont_title"])
    title =list(art_train[title]) #use user input variable
    path = "/".join(file_path.split("/")[:-1])  
    sent_embeddings, umap_data, predictions, probabilities = sent_embed_dr_cl(content, path)

    df = pd.DataFrame(title, columns=["Title"])
    df["predictions"] = predictions   

    with open(f"{path}/top_10_words.json", "r") as f:
        top_10_words = json.load(f)    

    top_words = []
    for pred in predictions:
        check = []
        for word in top_10_words[str(pred)]:
            check.append(word[0])
        top_words.append(check)
    df["Top Words"] = top_words
    
    var = input("Do you want to store new article's:\n\t[1]Sentence Transformer Embeddings\n\t[2]UMAP Dimensionality Reduced Embeddings\n\t[3]Articles with predictions\n\t[4]All\n\t[5]Don't save any\nYou can Select multiple options comma separated (e.g.1,2):")

    path = "/".join(file_path.split("/")[:-1]) 
    for v in var.split():           
        if int(v) == 1 or int(v) == 4:
            torch.save(sent_embeddings, f"{path}/1.new_articles_sentence_embeddings.pt")
        if int(v) == 2 or int(v) == 4:
            torch.save(umap_data, f"{path}/2.new_articles_sentence_dim_reduced_embeddings.pt")
        if int(v) == 3 or int(v) == 4:
            df.to_csv(f"{path}/3.predictions_file.csv")
        else:
            print("Thank you for using Neural Topic Modelling! :))")
            return
    
    print(f"File(s) have been saved at {path}")        
    return 

if __name__ == "__main__":
    main()