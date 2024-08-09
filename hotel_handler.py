# Importing the libraries
import os
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import shutil
import re

# Cache the data into the Hugging Face environment and disable the symlink warnings
os.environ['TRANSFORMERS_CACHE'] = 'C:/Users/jessi/.cache/huggingface'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Constants - max length of the output and calling the CSV data as a new name for inputs and also adding a relevance threshold
MAX_TEXT_LENGTH = 512
YELP_DATA = "C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Datasets/yelp_data0608.csv"
RELEVANCE_THRESHOLD = 0.7

# Load model from cache
model_name = 'sentence-transformers/all-mpnet-base-v2'
try:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully from cache!")
except Exception as e:
    print(f"Error loading model: {e}")

class Hotel_Data_Handler:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = embedding_model
        self.print_csv_headers(YELP_DATA)
        self.yelp_vector_store = self.create_csv_vector_store(YELP_DATA)

    @staticmethod
    def preprocess_text(text):
        """Preprocess the input text by removing special characters and specific words."""
        # Remove special characters
        text = re.sub(r"[^a-zA-Z0-9\s.!]", "", text)
        
        # Remove specific words
        words_to_remove = ['false', 'nan']
        for word in words_to_remove:
            text = text.lower().replace(word, '')
        
        return text

    def create_csv_vector_store(self, csv_path):
        try:
            df = pd.read_csv(csv_path, header=0, encoding='utf-8').sample(n=10000, random_state=1)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, header=0, encoding='ISO-8859-1')
        df = self.preprocess_data(df)
        hf_embeddings = self.create_hf_embeddings()
        text_chunks = df['extra_description'].tolist()
        try:
            vectorstore = FAISS.load_local('faiss_yelp_index2', hf_embeddings, allow_dangerous_serialization=True)
        except Exception as e:
            print(f"Error loading FAISS vector store: {e}")
            vectorstore = self.create_vector_store_from_texts(text_chunks, hf_embeddings)
            vectorstore.save_local("faiss_yelp_index2")
        return vectorstore

    def print_csv_headers(self, csv_path):
        """Prints out the headers of the CSV file."""
        try:
            df = pd.read_csv(csv_path, nrows=0)  # Read only the header row
            print(f"Headers in {csv_path}: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    def create_hf_embeddings(self):
        model_name = self.embedding_model
        hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return hf_embeddings

    def create_extra_description(self, row):
        try:
            description = (
                f"{row['city']} is a lovely place to visit and I'd surely recommend {row['name']}. "
                f"If {row['categories']} activities are your thing, this is the best place to visit! "
                f"The rating for this establishment is {row['stars']} so im sure you would enjoy it."
            )
        except KeyError as e:
            missing_column = e.args[0]
            description = f"Information missing for {missing_column}. "
            for col in ['city', 'state', 'attributes', 'categories', 'is_open', 'hours', 'stars']:
                if col in row:
                    description += f"{col.capitalize()}: {row[col]}. "
            print(f"Warning: {missing_column} column is missing.")
        return description

    def preprocess_data(self, df):
        df['extra_description'] = df.apply(self.create_extra_description, axis=1)
        df['extra_description'] = df['extra_description'].apply(self.preprocess_text)  # Preprocess text
        return df

    def create_vector_store_from_texts(self, text_chunks, embeddings):
        """
        Create a FAISS vector store from given text chunks and embeddings.
        :param text_chunks: List of text chunks to be embedded and indexed.
        :param embeddings: Embeddings model to use.
        :return: FAISS vector store.
        """
        vectorstore = FAISS.from_texts(text_chunks, embeddings)
        return vectorstore
    
    def vector_search(self, query, faiss_vector_store, top_k=4):
        relevant_docs = faiss_vector_store.similarity_search(query, k=top_k)
        return {"docs": relevant_docs, "are_relevant": True}
    
    def vector_search_yelp_csv(self, query):
        return self.vector_search(query, self.yelp_vector_store)

# Instantiate and use the Data_Handler
if __name__ == "__main__":
    data_handler =  Hotel_Data_Handler()
    query = "best food in Philadelphia"
    results = data_handler.vector_search_yelp_csv(query)
    print(results)
