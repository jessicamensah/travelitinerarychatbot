import os
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import shutil
# Set environment variables
os.environ['TRANSFORMERS_CACHE'] = 'C:/Users/jessi/.cache/huggingface'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# Constants
MAX_TEXT_LENGTH = 512
HOTEL_DATA = "C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Datasets/hotel_data.csv"
YELP_DATA = "C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Datasets/yelp_data.csv"
RELEVANCE_THRESHOLD = 0.7
# Load model from cache
model_name = 'sentence-transformers/all-mpnet-base-v2'
try:
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully from cache!")
except Exception as e:
    print(f"Error loading model: {e}")
class Data_Handler:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = embedding_model
        self.print_csv_headers(YELP_DATA)  # Print headers for debugging
        self.yelp_vector_store = self.create_csv_vector_store(YELP_DATA)
        self.hotel_vector_store = self.load_existing_vector_store("faiss_index")

    def load_existing_vector_store(self, index_path):
        """Load an existing FAISS vector store, or create a new one if not found."""
        hf_embeddings = self.create_hf_embeddings()
        try:
            vectorstore = FAISS.load_local(index_path, hf_embeddings, allow_dangerous_serialization=True)
            print(f"FAISS vector store loaded from {index_path}")
        except FileNotFoundError:
            print(f"FAISS vector store not found at {index_path}, creating a new one from Yelp data...")
            df = pd.read_csv(YELP_DATA, header=0, encoding='utf-8')
            df = self.preprocess_data(df)
            vectorstore = self.create_vector_store_from_texts(df['extra_description'].tolist(), hf_embeddings)
            vectorstore.save_local(index_path)
        except Exception as e:
            print(f"Error loading FAISS vector store from {index_path}: {e}")
            vectorstore = None
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
                f" Wow! {row['cityName']} is very beautiful and I'd surely recommend {row['HotelName']}. "
                f"It has {row['HotelFacilities']} and if you like {row['Attractions']}, this is the best place to stay! "
                f"The rating for this establishment is {row['HotelRating']}. In fact, here is the direct link! {row['HotelWebsiteUrl']}"
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
        return df
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
#        is_relevant = self.are_docs_relevant_to_query(query, relevant_docs)
        return {"docs": relevant_docs, "are_relevant": True}
    
    def yelp_vector_search(self, query, top_k=4):
        return self.vector_search(query, self.yelp_vector_store, top_k)
    
    def hotel_vector_search(self, query, top_k=4):
        return self.vector_search(query, self.hotel_vector_store, top_k)
    
 #   def are_docs_relevant_to_query(self, query, relevant_docs):
 #       hf_embeddings = self.create_hf_embeddings()
 #       query_embedding = hf_embeddings.embed(query)
        
        for doc in relevant_docs:
            doc_embedding = hf_embeddings.embed(doc.page_content)
            similarity_score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            if similarity_score >= RELEVANCE_THRESHOLD:
                return True 
        return False
    def vector_search_hotel_csv(self, query):
        return self.vector_search(query, self.hotel_vector_store)
    
    def vector_search_yelp_csv(self, query):
        return self.vector_search(query, self.yelp_vector_store)
    
# Instantiate and use the Data_Handler
if __name__ == "__main__":
    data_handler = Data_Handler()
    query = "best hotels in New York"
    results = data_handler.vector_search_yelp_csv(query)
    print(results)
