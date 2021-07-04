#!/bin/bash

#====================================== UBC - MDSCL ===================================================================
#
#          FILE: cmd_train.sh
#
#         USAGE: sh <right_path_to_the_file>/cmd_train.sh
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
#                VAR_6 = Selection of a gram for topic words generation -> 1:unigram ; 2:unigram & bigram [Default: 2]
#
#=======================================================================================================================

VAR_1="/Users/gurpreetbedi/Downloads/article.json"
VAR_2="sent_embeddings"
VAR_3="/Users/gurpreetbedi/Downloads/output_path/"
VAR_4="content"
VAR_5="contentTitle"
VAR_6=1 

#User Sentence embeddings to perform clustering
python3 cmd_NTM_train.py "$VAR_1" "$VAR_2" "$VAR_3" "$VAR_4" "$VAR_5" "$VAR_6"
