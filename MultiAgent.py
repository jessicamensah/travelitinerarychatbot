import gradio as gr  # type: ignore
from langchain_community.llms import Ollama
from data_handler import Data_Handler
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from io import StringIO
from html.parser import HTMLParser


chat_history = [] # show the chat history

# Maximum number of follow-up questions
MAX_QUESTIONS = 2

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class TravelAssistant:  #
    def __init__(self, hf_embeddings=None, hotel_index_path='C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Final Project/faiss_hotel_index', yelp_index_path='C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Final Project/faiss_yelp_index2'):
        self.llm = Ollama(model="phi3:3.8b")

        # Initialize embeddings and load index for vector search
        if hf_embeddings is None:
            hf_embeddings = SentenceTransformer('all-mpnet-base-v2')

        self.question_count = 0
        self.data_handler = Data_Handler(hotel_vector_store_path=hotel_index_path, yelp_vector_store_path=yelp_index_path) #the data handler has been called into the Multi Agent files and all it functions will work here.

        print('vector_store')

    # Function to count words in a string
    def word_count(self, s):
        return len(s.split())

    # Function to limit the prompt to a certain number of words
    def limit_words(self, prompt, max_words):
        words = prompt.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + "..."
        return prompt
    
    def analyse_convo_and_recommend(self, chat_history_text):
        summary_of_chat_history = self.convert_chat_history_to_summary(chat_history_text) # this code makes the chat history into a summary
        print(summary_of_chat_history)

        search_results = self.data_handler.yelp_vector_search(summary_of_chat_history) # its given in to the vectors to search the top k similarities
        print(search_results)

        response = self.make_recommendations_to_user(summary_of_chat_history, search_results) # the recommendation is ready, 
        print(response)

        return response

    def convert_gradio_history_to_text(self, message, gradio_history):
        """
        converts the gradio chat history into one string
        parameters:
            Message: the last user message
            gradio_history:  list of two-element lists of the form [[user_message, bot_message]]

        returns text_history:str
        """
        text_history = ""
        if gradio_history is not None:
            
            for user_message, ai_message in gradio_history:
                text_history = text_history + "\n User: " + user_message + "\n You: " + ai_message

        text_history = text_history + "\n User: " + message

        return text_history

    def travel_convo(self, message, history,):
        response = "unable to generate response, refresh application"
        history_text = self.convert_gradio_history_to_text(message, history)
        try:
            # Check if we have already asked the maximum number of questions
            if self.question_count >= MAX_QUESTIONS:
                print('there is a maximum amount of questions')
                return self.analyse_convo_and_recommend(history_text)
            
            # Check if there's enough information for a vector search
            if self.is_enough_info_for_search(history_text):
                print('the search is ready to start')
                response = self.analyse_convo_and_recommend(history_text)
                

                # Generate a recommendation based on the search results and conversation history
                return response
            else:
                print('the search is not ready to start')
                self.question_count = self.question_count + 1
                return "I need more information to provide a recommendation."
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return response

    def is_enough_info_for_search(self, chat_history_text):
        # The purpose of this function is to determine whether we have enough information from the user to do a vector search
        enough_information = False
        
        prompt = (
            "You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer. \
            I'm going to show you a travel conversation between a travel agent and the customer. \
            Based on the chat history, check if the customer has communicated the country and city they'd like to go to. \
            If there's enough information, reply 'yes'. If there isn't enough information, reply 'no'. \
            Here is the chat history: " + chat_history_text
        )
        
        
        # Send the prompt to the LLM
        # Read the LLM's response; if the response is "no," set enough_information to False; if "yes," set it to True
        
        response = self.llm.invoke(prompt) # sending the prompt to the llm - check if its a string

        if "yes" in response.lower():
            enough_information = True

        return enough_information

    def convert_chat_history_to_summary(self, text_chat_history):
        
        
        prompt = (
            "You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer. \
            I'm going to show you a travel conversation between a travel agent and the customer. \
            Based on the chat history, create a one line summary of what the customer is looking for.\
            Here is the chat history: " + text_chat_history
        )
        
        
        # Send the prompt to the LLM
        # Read the LLM's response; if the response is "no," set enough_information to False; if "yes," set it to True
 
        response = self.llm.invoke(prompt) # sending the prompt to the llm - check if its a string

        return response
    
    def make_recommendations_to_user(self, one_line_summary, search_results):
#         prompt = (
#    # f"You are a Travel Itinerary Agent. Based on the user's requests: {one_line_summary}, and the search results: {search_results['docs']}, make relevant travel recommendations."
# )

        prompt = (
            "You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer. \
            This is a summary of what the user is looking for:" + str(one_line_summary) +
            "Based on the relevant search results below, make relevant travel recommendations to the user. Here are the search results:"  + str(strip_tags(search_results["docs"][0].page_content)) # we are acessing the value behind the docs key # we are only going to use the docs instead of the full search results
        ) 
        print(prompt)
        
        # #             

        # Send the prompt to the LLM
         # when you send the message to llm, how long you give before it timesout
        response = self.llm.invoke(prompt) # sending the prompt to the llm - check if its a string
        # response = 'Based on your query, this it what I can recommend' + str(strip_tags(search_results["docs"][0].page_content))
        # print()
        
        print(response)
        return response
