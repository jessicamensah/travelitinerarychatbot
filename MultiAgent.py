import gradio as gr  # type: ignore
from langchain_community.llms import Ollama
from data_handler import Data_Handler
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the chat history
chat_history = []

# Maximum number of follow-up questions
MAX_QUESTIONS = 3

# Define the TravelAssistant class
class TravelAssistant:
    def __init__(self, index_file='index.faiss'):
        # Initialize embeddings and load index for vector search
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(index_file, self.embeddings)

    def vector_search(self, query):
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
        llm = Ollama(model="llama-7b")
        response = llm.invoke(prompt)

        # Post-processing check for hallucinations
        if "fabricated" in response or "assumption" in response:
            return "The response seems to contain hallucinations. Please provide more details."
        
        return response

# Function to count words in a string
def word_count(s):
    return len(s.split())

# Function to limit the prompt to a certain number of words
def limit_words(prompt, max_words):
    words = prompt.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + "..."
    return prompt

def travel_convo(message, history, question_count):
    try:
        # Check if we have already asked the maximum number of questions
        if question_count >= MAX_QUESTIONS:
            return "I have asked too many questions already. Please provide the required information."
        
        # Check if there's enough information for a vector search
        if is_enough_info_for_search(message, history):
            assistant = TravelAssistant()
            results = assistant.vector_search(history)
            
            # Use the results of the vector search in a prompt to ask the LLM for a recommendation
            prompt = (
                "Travel itinerary agent. Based on the chat history, provide recommendations or ask for more info. "
                "Chat history: " + history + "\n" + message
            )
            # Limit the prompt to a specific number of words
            prompt = limit_words(prompt, 100)  # Adjust the word limit as needed
            llm = Ollama(model="llama-7b")
            response = llm.invoke(prompt)
            
            # Post-processing check for hallucinations
            if "fabricated" in response or "assumption" in response:
                return "The response seems to contain hallucinations. Please provide more details."
            
            return response
        else:
            return "I need more information to provide a recommendation."
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def is_enough_info_for_search(message, history):
    # The purpose of this function is to determine whether we have enough information from the user to do a vector search
    enough_information = False
    # Append the latest message to the chat history list 
    chat_history = history.append(message)
    # Convert the historical list to a string/text format 
    chat_history_text = "\n".join(chat_history)
    
    prompt = (
        "You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer. "
        "I'm going to show you a travel conversation between a travel agent and the customer. "
        "Based on the chat history, check if the customer has communicated the country and city they'd like to go to. "
        "If there's enough information, reply 'yes'. If there isn't enough information, reply 'no'. "
        "Here is the chat history: " + chat_history_text
    )
    
    # Send the prompt to the LLM
    # Read the LLM's response; if the response is "no," set enough_information to False; if "yes," set it to True
    
    return enough_information

def gradio_interface(message):
    global chat_history
    
    # Append the new message to the chat history
    chat_history.append("User: " + message)
    history_text = "\n".join(chat_history)
    
    # Split the history and count the number of questions asked
    question_count = history_text.count("Question:")
    response = travel_convo(message, history_text, question_count)
    
    # Append the response to the chat history
    chat_history.append("Agent: " + response)
    
    return response

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, label="New Message"),
    outputs=gr.Textbox(),
    title="Travel Itinerary Agent",
    description="Ask the travel itinerary agent for recommendations. Provide your new message."
)

iface.launch(share=True)

