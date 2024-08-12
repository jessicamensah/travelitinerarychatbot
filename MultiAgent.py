import gradio as gr  # type: ignore
from langchain_community.llms import Ollama
from data_handler import Data_Handler
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


chat_history = [] # show the chat history

# Maximum number of follow-up questions
MAX_QUESTIONS = 3

class TravelAssistant:  #
    def __init__(self, index_file='C:/Users/jessi/OneDrive/Documents/Masters/Dissertation/Disso Project/Final Project/faiss_hotel_index'):
        # Initialize embeddings and load index for vector search
        self.embeddings = OpenAIEmbeddings(openai_api_key="sk-") # for access and to embed the vector
        self.vector_store = FAISS.load_local(index_file, self.embeddings, allow_dangerous_deserialization=True)
        self.question_count = 0
        self.data_handler = Data_Handler()
        

    def vector_search(self, query): #function to create a similarity search
        # Perform vector search on user input
        results = self.vector_store.similarity_search(query) 
        return results

        # Use the results of the vector search in a prompt to ask the LLM for a recommendation
        prompt = (
            "Travel itinerary agent. Based on the chat history, provide recommendations or ask for more info. "
            "Chat history: " + query + "\n"
        )
        # Limit the prompt to a specific number of words
        prompt = limit_words(prompt, 100)  # Adjust the word limit as needed
        llm = Ollama(model="phi3:3.8b") # the llm being used
        response = llm.invoke(prompt) # adding the prompt to the llm

        # Post-processing check for hallucinations
        if "fabricated" in response or "assumption" in response:
            return "The response seems to contain hallucinations. Please provide more details."
        
        return response

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
        one_line_summary = self.convert_chat_history_to_summary(chat_history_text)
        search_results = self.data_handler.yelp_vector_search(one_line_summary)
        response = self.make_recommendations_to_user(one_line_summary, search_results)

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

    def travel_convo(self, message, history):
        history_text = self.convert_gradio_history_to_text(message, history)
        try:
            # Check if we have already asked the maximum number of questions
            if self.question_count >= MAX_QUESTIONS:
                return self.analyse_convo_and_recommend(history_text)
            
            # Check if there's enough information for a vector search
            if self.is_enough_info_for_search(history_text):
                response = self.analyse_convo_and_recommend(history_text)
                

                # Generate a recommendation based on the search results and conversation history
                return response
            else:
                self.question_count = self.question_count + 1
                return "I need more information to provide a recommendation."
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")

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
        
        llm = Ollama(model="phi3:3.8b") 
        response = llm.invoke(prompt) # sending the prompt to the llm - check if its a string

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
        
        llm = Ollama(model="phi3:3.8b") 
        response = llm.invoke(prompt) # sending the prompt to the llm - check if its a string

        return response
    
    def make_recommendations_to_user(self, one_line_summary, search_results):
        
        
        prompt = (
            "You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer. \
            This is a summary of what the user is looking for: + one_line_summary \
            Based on the relevant search results below, make relevant travel recommendations to the user \
            Here are the search results: " + str(search_results)
        )
        
        
        # Send the prompt to the LLM
        
        llm = Ollama(model="phi3:3.8b") 
        response = llm.invoke(prompt) # sending the prompt to the llm - check if its a string

        return response
