import torch
import json
import sys
from collections import defaultdict
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import umap
import hdbscan
from scipy.stats import kendalltau
from typing import List
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import os
from os import path
from scipy.cluster.hierarchy import linkage
import plotly.figure_factory as ff

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import string 
import time
import gensim
from gensim import corpora
import glob


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import re
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Union, Mapping, Any
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import MinMaxScaler

import plotly.express as px
import plotly.graph_objects as go


sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0, "height":20, "width":20}
sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
stop_words = stopwords.words('english')
stop_words.extend(['am','really','they','go','get','we','me','would','like','great', "try", "must", "things","did","include", "thank", "tell", "thanks", "inside", "examples", "become",
                   "using","indeed"]) 

pd.set_option('max_colwidth', 400)

viridis = cm.get_cmap('viridis', 20)

from sentence_transformers import SentenceTransformer, util



def heatmap_visualization(embeddings, topics, top_n_words, width = 800, height = 800):
    """
    Create a heatmap visualization using Sentence Transformer embeddings and topics generated
    """
    topics_list = topics
    topics_lookup = {topic:i for i, topic in enumerate(topics_list)}
    indices = np.array([topics_lookup[topic] for topic in topics_list])
    embeddings = embeddings[indices]
    distance_matrix = cosine_similarity(embeddings)

    named_labels = [[[str(topic), None]] + top_n_words[topic] for topic in topics_list]
    named_labels = ["_".join([label[0] for label in labels[:4]]) for labels in named_labels]
    named_labels = [label if len(label) < 30 else label[:27] + "..." for label in named_labels]

    fig = px.imshow(distance_matrix, labels=dict(color="Similarity Score"), x=named_labels, y=named_labels, color_continuous_scale='GnBu'
                    )

    fig.update_layout(
        title={
            'text': "<b>Indeed Article's Similarity Matrix", 'y': .95, 'x': 0.55, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22,color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text='Trend')
    
    return fig, distance_matrix


