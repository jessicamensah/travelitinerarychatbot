import numpy as np
import pandas as pd
import openai
import os
import faiss
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import csv

MAX_TEXT_LENGTH = 512
HOTEL_DATA = "C:\Users\jessi\OneDrive\Documents\Masters\Dissertation\Disso Project\Datasets\hotel_data.csv"  # Change this to the actual path
YELP_DATA = "C:\Users\jessi\OneDrive\Documents\Masters\Dissertation\Disso Project\Datasets\yelp_data.csv"  # Change this to the actual path
RELEVANCE_THRESHOLD = 0.7



class Data_Handler:

    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):

        self.embedding_model = embedding_model
        self.yelp_vector_store = self.create_csv_vector_store(YELP_DATA)
        self.hotel_vector_store = self.create_csv_vector_store(HOTEL_DATA)


    def create_hf_embeddings(self):
        model_name =  self.embedding_model
        hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return hf_embeddings

    def create_extra_description(self, row):
        description = (
            f"The name of the city is {row['city']} which is situated in {row['state']}. "
            f"It has {row['attributes']} and if you like {row['categories']}, this is the best destination! "
            f"This activity is {row['is_open']} from {row['hours']}. "
            f"The rating for this establishment is {row['stars']}."
        )
        return description

    def preprocess_data(self, df):
        df['extra_description'] = df.apply(self.create_extra_description, axis=1)
        return df

    def create_csv_vector_store(self, csv_path):
        df = pd.read_csv(csv_path, header=0)
        df = self.preprocess_data(df)
        hf_embeddings = self.create_hf_embeddings()

        documents = [Document(page_content=row['extra_description'], metadata=row.to_dict()) for idx, row in df.iterrows()]
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        try:
            vectorstore = FAISS.load_local('faiss_doe_index_constitution', hf_embeddings)
        except:
            vectorstore = FAISS.from_documents(texts, hf_embeddings)
            vectorstore.save_local("faiss_doe_index_constitution")

        return vectorstore

    def vector_search(self, query, faiss_vector_store, top_k=4):
        relevant_docs = faiss_vector_store.similarity_search(query, k=top_k)
        is_relevant = self.are_docs_relevant_to_query(query, relevant_docs)
        return {"docs": relevant_docs, "are_relevant": is_relevant}

    def are_docs_relevant_to_query(self, query, relevant_docs):
        hf_embeddings = self.create_hf_embeddings()
        query_embedding = hf_embeddings.embed(query)
        
        for doc in relevant_docs:
            doc_embedding = hf_embeddings.embed(doc.page_content)
            similarity_score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            if similarity_score >= RELEVANCE_THRESHOLD:
                return True 

        return False

    def vector_search_hotel_csv(self, query):
        return self.processor.vector_search(query, self.hotel_data)

    def vector_search_yelp_csv(self, query):
        return self.processor.vector_search(query, self.yelp_data)




travel_data_handler = Data_Handler()
travel_data_handler
result = travel_data_handler.vector_search_yelp_csv("query about YELP data")
print(result)
