# Introduction for Neural Model Application

Neural Model application uses [Sentence Transformers model](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-12-v3/tree/main) for analyzing encoding the original dataset.

Neural Model application has been built with two methods:
 - Single command line: having separate command line scripts for sentence encoding, dimensionality reduction and document clustering. Useful for Batch operations.
 - Interactive method: taking user input to perform operations. 

This has been developed to Indeed option to choose either to perform Neural topic modelling.

## Single Command Line

### Related files:

#### Python Scripts:
1. `cmd_NTM_train.py`:
  - Python script used to create pre-trained sentence embeddings, perform dimensionality reduction and document clustering over existing articles.
2. `cmd_NTM_make_preds.py`:
  - Python script used to create pre-trained sentence embeddings, perform dimensionality reduction and document clustering from previously stored UMAP and HDBSCAN model over new set articles of articles.

### Shell Scripts
3. `cmd_train.sh`:
  - calls the `cmd_NTM_train.py` and has self-customized arguments enabled which can be updated durng runtime in the file.
4. `cmd_new_articles_sentence_embeddings.sh`:
  - calls the `cmd_NTM_make_preds.py` and has self-customized arguments enabled which can be updated durng runtime in the file.
5. `cmd_new_articles_umap_embed_pred_final.sh`:
  - calls the `cmd_NTM_make_preds.py` and has self-customized arguments enabled which can be updated durng runtime in the file.

Step-by-step instructions on executing Neural model application. 

### Execution steps for Training
1. Introduction on the arguments in the `cmd_trian.sh` file:  
   Parameters include: 
    - `VAR_1` = Input New articles file path.  
    - `VAR_2` = Option to extract (for eg: sent_embeddings or umap_embeddings).  
    - `VAR_3` = Output directory for the embeddings to be stored.  
    - `VAR_4` = Name of the content column in source which is used for generate embeddings.  
    - `VAR_5` = Name of the title column in source which is used for generate embeddings.  
    - `VAR_6` = Selection of a gram for topic words generation -> 1:unigram ; 2:unigram & bigram [Default: 2].  
    
2. After editing the desired arguments, and specify the right paths inside the `cmd_trian.sh` file, Run with command line in your terminal or colab:  
    ```
    sh right_path_to_the_file/cmd_train.sh
    ```
    Results will be save as you customized in the shell script!  

### Execution steps for Prediction

1. `cmd_new_articles_sentence_embeddings.sh`: This would help users generate sentence transformers embeddings for new articles.
   
   Parameters include:   
    - `VAR_1` = Input New articles file path.  
    - `VAR_2` = Option to extract (for eg: sent_embeddings or umap_embeddings).   
    - `VAR_3` = Output directory for the embeddings to be stored.  
    - `VAR_4` = Name of the content column in source which is used for generate embeddings.   
    - `VAR_5` = Name of the title column in source which is used for generate embeddings.     
  

2. `cmd_new_articles_umap_embed_pred.sh`:  This would help users to generate umap embeddings and predict clustering for new articles from previously saved models.
 
   Parameters include:
    - `VAR_1` = Input New articles file path
    - `VAR_2` = Option to extract (for eg: sent_embeddings or umap_embeddings)
    - `VAR_3` = Output directory for the embeddings to be stored
    - `VAR_4` = Name of the content column in source which is used for generate embeddings
    - `VAR_5` = Name of the title column in source which is used for generate embeddings
    - `VAR_6` = Directory of the UMAP folder where umap model is saved from the previous cmd_train.sh execution
    - `VAR_7` = Directory of the HDBSCAN folder where umap model is saved from the previous cmd_train.sh execution
    - `VAR_8` = Select the saved UMAP model to generate new article dimensionality reduced embeddings and predictions
    
3. After editing the desired arguments, and specify the right paths inside the shell file(s), Run with command line in your terminal or colab:   
    ```
    sh right_path_to_the_file/cmd_new_articles_sentence_embeddings.sh
    ```
    or:
    ```
    sh right_path_to_the_file/cmd_new_articles_umap_embed_pred_final.sh
    ```
    
## Interative

Step-by-step instructions on executing Neural model application. 

### Related files:  
1. `import_lib_n_functions.py`: This file will import required libraries and functions.
2. `pre_process.py`: This file will pre-process the provided raw data.
3. `NTM_training.py`: This file will train the model.
4. `NTM_make_predictions.py`: This file will do the predictions.

### Execution steps for Training

1. Before playing around with these python files, please making sure:  
    a. install:  
        ```
        !pip install sentence_transformers
        !pip install umap-learn
        !pip install hdbscan
        ```
            
    b. Have your raw data file ready in Json format with proper file path. (Do not forget to mount you drive if you run it on Google Drive!)
        for example: 
        ```
        /content/gdrive/MyDrive/indeed/Copy of article.json
        ```

2. Run the following command line in the terminal (Make sure the py file path is correct.):
    ```
    !python3 /content/gdrive/MyDrive/indeed/NTM_training.py
    ```
     
3. Copy and paste the raw data file path when the terminal askes for it:  
    ```
    Loading Libraries...Just a moment please!...
    2021-06-21 19:55:05.757217: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    Please provide the full path for the article.json file for training:
    /content/gdrive/MyDrive/indeed/Copy of article.json
    ```
    
4. Then it will confirm that you enter the path you want, and ask if you want to proceed (Press Y or Yes to proceed):  
    ```
    You have provided: /content/gdrive/MyDrive/indeed/Copy of article.json
    Can we proceed [Y for Yes]?
    y
    ```
    
5. It will tell you that file exists with a big smile face :)), and then start loading the data you provided. It soppose to be quick for this step. Now you need to provide the article content and title column names from your data (in this case is `content` and `contentTitle`).
    ```
    File exists! :))
    Loading file now...
    Please content column name:
    content
    Please title columns name:
    contentTitle
    ```

6. After executing step 5, you will see what steps exactly are excuting with the progress bar. This progress will takes less than 3 minutes depends on how big the data you put in. 
    ```
    Pre-processing the dataset...
    Finalizing the dataset...
    File received with size: (13639, 8)
    Generating Sentence Transformer Embeddings and performing dimensionality reduction for proper clustering...
    Batches: 100% 427/427 [02:40<00:00,  2.65it/s]
    ```
    
7. See warnings like this? 
    ```
        huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    ```
    It is all good, it is the library we used from `huggingface` that inform you that you can disable the parallelism to avoid deadlocks..
    
8. Now you will see this:
    ```
    Sentence Transformer embeddings received of shape: (13639, 384)
        UMAP Reduced embeddings received of shape: (13639, 5)
        Number of HDBSCAN Clustered labels generated: 171        
    ```
    It generated the embedding shapes and the reduced UMAP embedding shape, and the clustering done by HDBSCAN.
    
    Now you want to choose which ngram you would like to use for topic words generation, currently we support two options:
    ```
    Semantically assigning cluster labels to the documents..
    What ngram range would you like to be used for topic words generation
        1 : unigram
        2 : bigram
    1
    Extracting Top most coherent words associated with semantically aligned documents...
    ```
    So imagine you select first option, which is asking to use `unigram`. 
    
9. Now, every steps for training part is done! You just need to choose what is the content you want to save in your local (or drive if you are using Google Colab).
    ```
    Do you want to store:
        [1]Sentence Transformer Embeddings
        [2]UMAP Dimensionality Reduced Embeddings
        [3]HDBSCAN clustered labels
        [4]Save Indeed Intertopic distance graph
        [5]Topics Dendrogram
        [6]All
        [7]Don't save anything
    You can Select multiple options comma separated (e.g.1,2):6
    File(s) have been save at /content/gdrive/MyDrive/indeed
    ```
    Imagine you chose option 6, which saved all the results. And then there should be a line appearing after that tells you where those files are saved. 
    
    
### Execution steps for Predicting
1. Run the `NTM_make_prediction.py` file (make sure the path is right):
    `!python3 /content/gdrive/MyDrive/indeed/NTM_make_predictions.py`
2. The terminal will show up something like following, and ask you to put the path of the new articles that you want to predict on.
    ```
    Loading Libraries...Just a moment please!...
    2021-06-21 19:49:54.915947: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    Please provide the full path for the article.json file for making predictions from the saved model:
    /content/gdrive/MyDrive/indeed/Copy of article_20210503_to_20210525.json
    ```
3. Same as previous step, we can choose the columns you want to proceed your predictions:
    ```
    Can we proceed [Y for Yes]?y
    File exists! :))
    Loading file now...
    Please content column name:
    content
    Please title columns name:
    contentTitle
    ```
4. We also have options for users to decide which to save:
    ```
    Do you want to store new article's:
        [1]Sentence Transformer Embeddings
        [2]UMAP Dimensionality Reduced Embeddings
        [3]Articles with predictions
        [4]All
        [5]Don't save any
    You can Select multiple options comma separated (e.g.1,2):4
    File(s) have been saved at /content/gdrive/MyDrive/indeed
    ```
**Note: Before making predictions, make sure the training has been executed as in order to make predictions the prediction scripts references the models saved from previous training run. If you are cloning this repo make sure all the files from `/src/` and `/app_scripts` are placed in the same folder.**
