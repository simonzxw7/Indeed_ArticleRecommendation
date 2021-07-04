print("Loading Libraries...Just a moment please!...")

from import_lib_n_functions import *

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

from os import path

def file_path_check(option):
    """
    Check if the article belongs to training or test set.
    """       
    if option=="train":
        file_path = input("Please provide the full path for the article.json file for training:\n")    
    else:
        file_path = input("Please provide the full path for the article.json file for making predictions from the saved model:\n")            
    
    print(f"You have provided: {file_path}")
    var = input("Can we proceed [Y for Yes]?")
    if var.lower() == "y":
        if str(path.exists(file_path)):
            print("File exists! :))")  
            return file_path
        else:
            raise ValueError("File path is not correct :((")
    else:
        raise ValueError("Please enter the correct path and enter 'Y' for Yes")

def check_col_names(df):
    """
    Check the colomn name of the dataframe.
    """        
    content = input("Please content column name (article.json has column named content):\n")    
    try:
        content in df.columns
    except KeyError:
        print(f"Please provide correct content column name, {content} does not exist in the dataset")
        
    title = input("Please title columns name (article.json has column named contentTitle):\n")          
    try:
        content in df.columns
    except KeyError:
        print(f"Please provide correct content column name, {content} does not exist in the dataset")        

    return content, title

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

def train_pre_process(file_path):
    """
    Call functions and implement pre-processing for training dataset.
    """        
    file_path = file_path_check("train")
    if file_path:
        print("Loading file now...")
        art_train = load_data(file_path)
        content, title = check_col_names(art_train)
        print("Pre-processing the dataset...")
        rltd_arts, art_train = get_corpus(art_train, content)
        print("Finalizing the dataset...")
        art_train["cont_title"] = art_train[title] + " " +art_train[content]
    return art_train, file_path, title

def predict_pre_process(file_path, content, title):
    """
    Call functions and implement pre-processing for test dataset.
    """    
    file_path = file_path_check("predict")
    if file_path:
        print("Loading file now...")
        art_train = load_data(file_path)
        content, title = check_col_names(art_train)        
        print("Pre-processing the dataset...")
        rltd_arts, art_train = get_corpus(art_train, content)
        print("Finalizing the dataset...")
        art_train["cont_title"] = art_train[title] + " " +art_train[content]
    return art_train, file_path, title

if __name__ == "__main__":
    train_pre_process(file_path, content, title)
    predict_pre_process(file_path, content, title)
    Sentence_Encoder()