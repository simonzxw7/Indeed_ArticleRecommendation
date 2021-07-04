from pre_process import *

def sent_embed_dr_cl(documents,path):
    """
    Get sentence embeddings, and coefficients from training the umap and hdbscan models.
    """    
    embedding_model = "msmarco-MiniLM-L-12-v3"
    device = "cuda" if torch.cuda.is_available else "cpu"
    sent_embeddings = Sentence_Encoder(embedding_model).embed_documents(documents, device=device, verbose=True)

    
    umap_model = umap.UMAP(n_neighbors=15,
                           n_components=5,
                           metric='cosine')


    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)

    umap_data = umap_model.fit_transform(sent_embeddings)
    cluster = hdbscan_model.fit(umap_data)

    dir = os.path.join(path,"Embeddings")
    if not os.path.exists(dir):
        os.mkdir(dir)
        os.mkdir(dir + "/train")
    print(f"Storing Sentence transformers embeddings at: {dir}/train")
    torch.save(sent_embeddings, f"{dir}/train/sentence_transformer_embeddings.pt")  
    print(f"Storing dimensionality reduced embeddings at: {dir}/train")    
    torch.save(umap_data, f"{dir}/train/umap_embeddings.pt")      
    
    dir = os.path.join(path,"UMAP")
    if not os.path.exists(dir):
        os.mkdir(dir)        
    joblib.dump(umap_model, f"{dir}/umap_model_{len(np.unique(cluster.labels_))}.sav")  
    
    dir = os.path.join(path,"HDBSCAN")
    if not os.path.exists(dir):
        os.mkdir(dir)            
    joblib.dump(hdbscan_model, f"{dir}/hdbscan_model_{len(np.unique(cluster.labels_))}.sav")

    return sent_embeddings, umap_data, cluster


class Neural_Topic_model:
    """
    Neural Topic Model is BERT based topic modeling technique which uses BERT embeddings and 
    topic based TF-IDF to create dense clusters allowing for easily interpretable topics
    during which keeping important words in the topic descriptions.
    """
    
    def __init__(self, documents, docs_df, docs_per_topic, count, tf_idf, top_words = 10, embedding_model = None, width=1000, height=1000, top_n_topics=None, ngram_range=(1,2)):
        """
        Define parameters of the neural topic model.
        """      
        if top_words > 20:
            raise ValueError("top_n_words should be lower or equal to 20. The preferred value is 10.")
        self.top_words = top_words 
        self.ngram_range = ngram_range   
        self.documents = documents     
        self.docs_df = docs_df
        self.docs_per_topic = docs_per_topic
        self.count = count
        self.tf_idf = tf_idf        
        self.top_10_words = self.top_words_per_topic(docs_per_topic, self.top_words)        
        self.topic_sizes = self.topic_sizes(docs_df)
        self.topics = self.top_10_words
        self.docs_length = len(documents)
        self.width = width
        self.height = height        
        self.top_n_topics = top_n_topics
  
    
    def top_words_per_topic(self, docs_per_topic, n=20):
        """
        Get the top words per topic
        """        
        words = self.count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = self.tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_10_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_10_words

    def topic_sizes(self, df):
        """
        Get the document counts clustered for all the topics.
        """
        topic_sizes = (df.groupby(['Topic']).Doc.count().reset_index().rename({"Topic": "Topic", "Doc": "Doc_Count"}, axis='columns')
                       .sort_values("Doc_Count", ascending=False))
        return topic_sizes
    
    def get_colors(self, topics, topic_list):
        """
        Get the color marker for the topics inside the topics list
        """          
        if topics == -1:
            marker = ["#B0BEC5" for _ in topic_list[1:]]
        else:
            marker = ["red" if topic == topics else "#B0BEC5" for topic in topic_list[1:]]
        return [{'marker_color': [marker]}]       
    
    def get_topic(self, topic):
        """
        Get a particular topic
        """        
        if topic in self.topics:
            return self.topics[topic]
        else:
            return False     
        
    
    def get_topic_freq(topic=None):
        """
        Get document count per topic 
        """       
        if isinstance(topic, int):
            return self.topic_sizes[topic]
        else:
            return pd.DataFrame(topic_sizes.items(), columns=['Topic', 'Doc_Count']).sort_values("Doc_Count",ascending=False)


    def topics_viz_data(self):
        """
        Creates Topic Visalization in 2D space. Each topic taking shape as the sice of the documents clustered in each topic
        """        
        topics_list = sorted(list(self.topics.keys()))
        topics_lookup = {topic:i for i, topic in enumerate(topics_list)}

        frequencies = [self.topic_sizes[self.topic_sizes["Topic"]==topic]["Doc_Count"].values[0] for topic in topics_list]
        words = [" | ".join([word[0] for word in self.get_topic(topic)[:5]]) for topic in topics_list]

        indices = np.array([topics_lookup[topic] for topic in topics_list])

        embeddings = self.tf_idf[indices]
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = umap.UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)

        df = pd.DataFrame({"x": embeddings[1:, 0], "y": embeddings[1:, 1],
                           "Topic": topics_list[1:], "Words": words[1:], "Size": frequencies[1:]})
        fig = self.topic_visualization_formatting(df, topics_list)
        return fig    

    def create_topic_embeddings(self, sent_embeddings):
        """
        Create topic embeddings using sentence transformer embeddings
        """
        topic_list = list(self.topics.keys())
        topic_list.sort()
        n=self.top_words

        topic_embeddings = []
        for i, topic in enumerate(topic_list):
            indexes = np.array(self.docs_df[self.docs_df["Topic"]==topic].index)
            topic_embeddings_ = np.average(sent_embeddings[indexes], axis=0)
            topic_embeddings.append(topic_embeddings_)
        return topic_embeddings

    
    def topics_dendrogram(self, sent_embeddings, orientation="left"):
        """
        Creates a Dendogram in topic space using sentence transformer embeddings
        """
        embeddings = self.create_topic_embeddings(sent_embeddings)
        embeddings = np.array(embeddings)
        
        topics_list = sorted(list(self.topics.keys()))
        topics_lookup = {topic:i for i, topic in enumerate(topics_list)}
        indices = np.array([topics_lookup[topic] for topic in topics_list])
        embeddings = embeddings[indices]

        # Create dendogram
        distance_matrix = 1 - cosine_similarity(embeddings)
        fig = ff.create_dendrogram(distance_matrix,
                                   orientation=orientation,
                                   linkagefun=lambda x: linkage(x, "ward"),
                                   color_threshold=1)

        # Create labels
        axis = "yaxis" if orientation == "left" else "xaxis"
        named_labels = [[[str(topics_list[int(x)]), None]] + self.get_topic(topics_list[int(x)])
                      for x in fig.layout[axis]["ticktext"]]
        named_labels = ["_".join([label[0] for label in labels[:4]]) for labels in named_labels]
        named_labels = [label if len(label) < 30 else label[:27] + "..." for label in named_labels]

        # Stylize layout
        fig.update_layout(plot_bgcolor="#ECEFF1",
                          template="plotly_white",
                          title={'text': "<b>Indeed's Article Clustering", "y": .95, "x": 0.5, "xanchor": 'center', "yanchor": "top", "font": dict(size=22, color="Black")},
                          width=self.width,
                          height=self.height,
                          hoverlabel=dict(bgcolor="white", font_size=16,font_family="Rockwell"),

        )

        # Stylize orientation
        if orientation == "left":
            fig.update_layout(yaxis=dict(tickmode="array",
                                         ticktext=named_labels))
        else:
            fig.update_layout(xaxis=dict(tickmode="array",
                                         ticktext=named_labels))
        return fig    

    def topic_visualization_formatting(self, df, topic_list):
        """
        Format topic visulization using the co-ordinates provided in the datafarme and list of topics
        """
        # Prepare figure range
        x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
        y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))  
        
        # Plot topics
        fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                         hover_data={"x": False, "y": False, "Topic": True, "Words": True, "Size": True})
        fig.update_traces(marker=dict(color="#7f91eb", line=dict(width=2, color='DarkSlateGrey')))

        # Update hover order
        fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                     "Words: %{customdata[1]}",
                                                     "Count: %{customdata[2]}"]))   

        # Create a topic selection slider
        steps = [dict(label=f"Topic {topic}", method="update", args=self.get_colors(topic, topic_list)) for topic in topic_list[1:]]
        sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

        # Stylize layout
        fig.update_layout(
                            title={'text': "<b>Indeed's Intertopic Distance", 'y': .95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22,color="Black")},
                            width=self.width,
                            height=self.height,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=14,
                                font_family="Rockwell"
            ),
            xaxis={"visible": False},
            yaxis={"visible": False},
            sliders=sliders
        )
        #Updating the axis with apropriate ranges
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)

        #Adding the shapes to the 
        fig.add_shape(type="line",
                      x0=sum(x_range)/2, y0=y_range[0], x1=sum(x_range)/2, y1=y_range[1],
                      line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="line",
                      x0=x_range[0], y0=sum(y_range)/2, x1=x_range[1], y1=sum(y_range)/2,
                      line=dict(color="#9E9E9E", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range)/2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(y=y_range[1], x=sum(x_range)/2, text="D2", showarrow=False, xshift=10)
        fig.data = fig.data[::-1]

        return fig        
      
    
def main():
    """
    Get the shape of sentence transformer embeddings, reduced embeddings from UMAP and the number of HDBSCAN clustered labels, together with the top topic words associated with aligned articles.
    """
    art_train, file_path, title = train_pre_process()
    print(f"File received with size: {art_train.shape}")

    content = list(art_train["cont_title"])
    title =list(art_train[title])
    path = "/".join(file_path.split("/")[:-1]) 
    
    print("Generating Sentence Transformer Embeddings and performing dimensionality reduction for proper clustering...")

    sent_embeddings, umap_data, cluster = sent_embed_dr_cl(content, path)

    print(f"""
        Sentence Transformer embeddings received of shape: {np.array(sent_embeddings).shape}
        UMAP Reduced embeddings received of shape: {umap_data.shape}
        Number of HDBSCAN Clustered labels generated: {len(np.unique(cluster.labels_))}        
    """)    
    
    print("Semantically assigning cluster labels to the documents..")    
    docs_df = pd.DataFrame(content, columns=["Doc"])
    docs_df['Title'] = title
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})   

    ngram = input("What ngram range would you like to be used for topic words generation\n\t1 : unigram\n\t2 : bigram\n")
    def doc_tf_idf(documents, doc_length, ngram_range=(1,1)):
        """
        topic based TF-IDF to create dense clusters allowing for easily interpretable topics
        """
        count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit(documents)
        terms = count.transform(documents).toarray()
        words = np.sum(terms, axis=1)
        tf = terms.T/words
        sum_terms = np.sum(terms, axis=0)
        idf = np.log(doc_length/sum_terms).reshape(-1, 1)
        tf_idf = tf * idf
        return tf_idf, count
    
    print("Extracting Top most coherent words associated with semantically aligned documents...")
    tf_idf, count = doc_tf_idf(docs_per_topic.Doc.values, doc_length=len(content), ngram_range=(1,int(ngram)))
    
    neural = Neural_Topic_model(content, docs_df, docs_per_topic, count, tf_idf)
    top_10_words = neural.top_words_per_topic(docs_per_topic,n=10)

    var = input("Do you want to save:\n\t[1]Indeed Intertopic distance graph\n\t[2]Topics Dendrogram\n\t[3]All\n\t[4]Don't save anything\nYou can Select multiple options  comma separated (e.g.1,2):")
    
    with open(f"{path}/top_10_words.json", "w") as f:
        json.dump(top_10_words, f)    
    dir = os.path.join(path,"Results")        
    for v in var.split():      
        if int(v) in np.arange(1,7):
            var = input("Please provide the output directory path:\n")     
            if int(v) == 1 or int(v) == 6:
                dir = os.path.join(path,"sent_embeddings")
                if not os.path.exists(dir):
                    os.mkdir(dir)                      
                torch.save(sent_embeddings, f"{dir}/sentence_embeddings.pt")                 
            if int(v) == 2 or int(v) == 6:
                dir = os.path.join(path,"umap_embeddings")
                if not os.path.exists(dir):
                    os.mkdir(dir)                                   
                torch.save(umap_data, f"{dir}/umap_embeddings_{len(np.unique(cluster.labels_))}")
            if int(v) == 3 or int(v) == 6:
                dir = os.path.join(path,"clustered_labels")
                if not os.path.exists(dir):
                    os.mkdir(dir)                                                   
                with open(f"{dir}/HDBSCAN_clustered_labels.npy","wb") as f:
                    np.save(f, cluster) 
            var_ = input(f"Is it ok to store results at path:\n{dir}\n\n If Yes enter Y or else provide the directory:\n")            
            if int(v) == 1 or int(v) == 3:
                fig = neural.topics_viz_data()
                if not os.path.exists(dir):
                    os.mkdir(dir)                 
                var = input(f"Is it ok to store results at path:\n{dir}\n\n If Yes enter Y or else provide the directory:\n")            
                if var_.lower() == "y":
                fig.write_html(f"{dir}/Indeeds_Intertopic_distance_{len(np.unique(cluster.labels_))}.html")
                else:
                    if not os.path.exists(var_):
                        os.mkdir(var_ + "/train")                         
                    fig.write_html(f"{dir}/Indeeds_Intertopic_distance_{len(np.unique(cluster.labels_))}.html")                        
            if int(v) == 2 or int(v) == 3:
                fig = neural.topics_dendrogram(sent_embeddings)
                if not os.path.exists(dir):
                    os.mkdir(dir)                     
                var = input(f"Is it ok to store results at path:\n{dir}\n\nIf Yes enter Y or else provide the directory:\n")            
                if var.lower() == "y":
                fig.write_html(f"{dir}/Indeed_Articles_dendrogram_{len(np.unique(cluster.labels_))}.html")
                else:
                    if not os.path.exists(var):
                        os.mkdir(dir + "/train")                                         
                    fig.write_html(f"{dir}/Indeed_Articles_dendrogram_{len(np.unique(cluster.labels_))}.html")
        else:
            print("Thank you for using Neural Topic Modelling! :))")
        
            return
    
    print(f"File(s) have been save at {dir}")    
    return 

if __name__ == "__main__":
    main()