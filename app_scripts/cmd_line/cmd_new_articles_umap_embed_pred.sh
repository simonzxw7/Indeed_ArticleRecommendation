#!/bin/bash

#====================================== UBC - MDSCL ========================================================================
#
#          FILE: cmd_new_articles_umap_embed_pred_final.sh
#
#         USAGE: sh <right_path_to_the_file>/cmd_new_articles_umap_embed_pred_final.sh
#
#   DESCRIPTION: This command file is used to generate umap embeddings for new articles and cluster predictions
# 
#        AUTHOR: Gurpreet Bedi
#    UNIVERSITY: University of British Columbia (UBC)
#       VERSION: 1.0
#       CREATED: 06/21/2021
#
#        PARAMS: VAR_1 = Input New articles file path
#                VAR_2 = Option to extract (for eg: sent_embeddings or umap_embeddings)
#                VAR_3 = Output directory for the embeddings to be stored
#                VAR_4 = Name of the content column in source which is used for generate embeddings
#                VAR_5 = Name of the title column in source which is used for generate embeddings
#                VAR_6 = Directory of the UMAP folder where umap model is saved from the previous cmd_train.sh execution
#                VAR_7 = Directory of the HDBSCAN folder where umap model is saved from the previous cmd_train.sh execution
#                VAR_8 = Select the saved UMAP model to generate new article dimensionality reduced embeddings and predictions
#
#==============================================================================================================================

VAR_1="/Users/gurpreetbedi/Downloads/pred_article.json"
VAR_2="umap_embeddings"
VAR_3="/Users/gurpreetbedi/Downloads/output_path/"
VAR_4="content"
VAR_5="contentTitle"
VAR_6="/Users/gurpreetbedi/Downloads/output_path/UMAP/"
VAR_7="/Users/gurpreetbedi/Downloads/output_path/HDBSCAN/"
VAR_8="umap_model_177.sav"


#UMAP embedidngs & predictions
python3 cmd_NTM_make_preds.py "$VAR_1" "$VAR_2" "$VAR_3" "$VAR_4" "$VAR_5" "$VAR_6" "$VAR_7" "$VAR_8"
