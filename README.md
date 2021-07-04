# Project-Indeed

Unsupervised Topic Modelling for Indeed Articles

## Summary

Indeed is one of the best job sites in the world, with thousands of users visiting each day, either to apply for a job or lookup for specific job-related articles. Since Indeed is an open-source website, it has both categories of users: registered and non-registered. We performed some NLP based analyses and without any supervised data with appropriate labels, forced us to look for unsupervised approach to extract topics, i.e. Unsupervised Topic Modelling.

## Project Goal

In this project, we are going to perform Unsupervised Topic Modelling by bulding an algorithm using non-neural and neural approach. Non-Neural approach would include bulding Gensim LDA Topic modelling and Scikit-learn Topic modelling, and neural approach would include applying state-of the art embeddings using pre-trained Sentence transformer model. Both the models were evaluation using the article to article link data provided by the client, which can be found [here](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/data)


## Folders

```
.
├── app_scripts
│   └── cmd_line
│       └── logs
│   └── interactive
|       └── logs
├── data
├── data_stats
├── documents
├── image
├── prediction
│   └── Neural
│   └── Gensim
├── src
│   └── Neural
│   └── Gensim


```

1. [app_scripts](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/app_scripts)
    - Contains neural model applicaiton command line direct and interactive scripts with respective reference logs. 
2. [data](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/data)
    - All the source data provided by the client would be foud in this location. Some of the files include:
      - [careerpathpage](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/data/careerpathpage.json.zip)
      - [categorypage](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/data/categorypage.json.zip)
      - [coverletter](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/data/coverletter.json.zip)
      - [article and link data](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/data/pred_article.json.zip)
      - [resumesamplepage](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/data/resumesamplepage.json.zip)
3. [data_stats](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/data_stats)
    - Include the Indeed dataset statistics
4. [documents](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/documents)
    - All the docuements including client documents and project plan could be found in this folder. 
5. [image](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/image)
    - Contains project related images.
6. [predictions](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/predictions/Neural)
    - All the files related to predictions made using neural and non-neural model could be found here
7. [src](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/src)
    - All the python source files could be at this location


## Libraries to be installed
In order to execute the `.ipynb` files for executing LongformerModel
```
!conda create --name longformer python=3.7
!conda activate longformer
!conda install cudatoolkit=10.0
!pip install git+https://github.com/allenai/longformer.git
```

For all executions make sure below libraries are already instaled

```
!pip install sentence_transformers
!pip install umap-learn 
!pip install hdbscan
!pip install -U plotly
!pip install ipynb
```
These could also be found in [requirements.txt](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/requirements.txt)


## Neural Model Applications:

Neural model application files and Steps for executing training and making predictions can be found [here](https://github.ubc.ca/gbedi90/Project-Indeed/tree/master/app_scripts)

Reference log can be found [here](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/app_logs/training_log.txt)


**Note: Before making predictions, make sure the training has been executed as in order to make predictions the prediction scripts references the models saved from previous training run. If you are cloning this repo make sure all the files from `/src/` and `/app_scripts` are placed in the same folder.**

Refernce log can be found [here](https://github.ubc.ca/gbedi90/Project-Indeed/blob/master/app_logs/prediction_log.txt)

## Contributors
- Gurpreet Bedi
- Simon Zheng
- Lisa Liu
