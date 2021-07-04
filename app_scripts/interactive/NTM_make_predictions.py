from pre_process import *

def sent_embed_dr_cl(documents, path):
    """
    Get sentence embeddings, umap embeddings, prediction on the test data together with the probabilities.
    """
    # Embedding model
    embedding_model = "msmarco-MiniLM-L-12-v3"
    device = "cuda" if torch.cuda.is_available else "cpu"
    new_embeddings = Sentence_Encoder(embedding_model).embed_documents(documents, device=device, verbose=True)
    
    dir = os.path.join(path,"Embeddings","test")
    if not os.path.exists(dir):
        os.mkdir(dir)     
    print(f"Storing Sentence transformers embeddings at: {dir}/test")        
    torch.save(new_embeddings, f"{dir}/sentence_transformer_embeddings.pt")      
    
    dir = os.path.join(path,"UMAP")      
    if not os.path.exists(dir):
        raise ValueError("The required UMAP model folder is not found. Please check if the folder exist or if the training has been exeucted for it to create the folder.")         
    umap_models = [i.split("/")[-1].split(".")[0] for i in glob.glob(dir + "/*.sav")]
    print(umap_models)
    var = input("Found these models, please choose which one to select:\n")
    if var in umap_models:
        umap_model = joblib.load(f"{dir}/{var}.sav")
        var = var.split("_")[-1]
        hdbscan_model = joblib.load(f"{os.path.join(path)}/HDBSCAN/hdbscan_model_{var}.sav")            
        new_umap_embeddings = umap_model.transform(new_embeddings)
        dir = os.path.join(path,"Embeddings","test")
        if not os.path.exists(dir):
            os.mkdir(dir)     
        torch.save(new_embeddings, f"{dir}/umap_embeddings.pt")          
        predictions, _ = hdbscan.approximate_predict(hdbscan_model, new_umap_embeddings)
        probabilities = hdbscan.membership_vector(hdbscan_model, new_umap_embeddings)     
    else:
        raise ValueError("Incorrect UMAP model, please try again and select the corect UMAP model")         
    
    return new_embeddings, new_umap_embeddings, predictions, probabilities, var
     
    
def main():
    """
    Performs test data pre-processing, transforms the data to reduced dimensionality and makes predictions abotu the cluster labels
    """
    art_train, file_path, title = predict_pre_process()
   
    content = list(art_train["cont_title"])
    title =list(art_train[title])
    path = "/".join(file_path.split("/")[:-1])  
    sent_embeddings, umap_data, predictions, probabilities, model_number = sent_embed_dr_cl(content, path)
    model_number = model_number.split("_")[-1]
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
    var = input("Do you want to store new article's:\n\t[1]Articles with predictions\n\t[2]Don't save\nPlease select required option:")
    path = "/".join(file_path.split("/")[:-1]) 
    dir = os.path.join(path,"Results")     
    for v in var.split():           
        if int(v) == 1 or int(v) == 4:
            if not os.path.exists(dir):
                os.mkdir(dir)
                os.mkdir(dir + "/test")               
            torch.save(sent_embeddings, f"{dir}/test/new_articles_sentence_embeddings.pt")
        if int(v) == 2 or int(v) == 4:
            if not os.path.exists(dir):
                os.mkdir(dir)
                os.mkdir(dir + "/test")            
            torch.save(umap_data, f"{dir}/test/new_articles_sentence_dim_reduced_embeddings.pt")
    if int(var) == 1:
        df.to_csv(f"{dir}/predictions_file_{model_number}.csv")
    else:
        print("Thank you for using Neural Topic Modelling! :))")
        return
    
    print(f"File(s) have been saved at {dir}")        
    return 

if __name__ == "__main__":
    main()