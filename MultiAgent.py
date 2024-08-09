import gradio as gr # type: ignore
from langchain_community.llms import Ollama
from data_handler import Data_Handler
#make a class 
# Initialize the chat history
chat_history = []
# Maximum number of follow-up questions
MAX_QUESTIONS = 3
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
        if is_enough_info_for_search(message, history):
            # do a vector search on our database using our information from our user
            # use the results of the vector search in a prompt ask our llm to make a recommendation

        # Construct a concise prompt
        prompt = (
            "Travel itinerary agent. Based on the chat history, provide recommendations or ask for more info. "
            "Chat history: " + history + "\n" + message
        )
        # Limit the prompt to a specific number of words
        prompt = limit_words(prompt, 100)  # Adjust the word limit as needed
        llm = Ollama(model="llama-7b") #phi3:3.8b
        response = llm.invoke(prompt)
        
        # Post-processing check for hallucinations
        if "fabricated" in response or "assumption" in response:
            return "The response seems to contain hallucinations. Please provide more details."
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def is_enough_info_for_search(message, history): # the purpose of this function is to determine whether we have enough information from the user to do a vector search  --- docstring - description of the functions
    enough_information = False
    #country and city needed
    chat_history = history.append(message) # add the latest message to the chat history list 
    chat_history_text = "\n".join(chat_history) # historical list in string/text format 
    prompt = ("You are a Travel Itinerary Agent. You are trying to figure out if you can make a good travel recommendation to a customer \\
            Im going to show you a travel conversation between a travel agent and the customer. \\
            Based on the chat history, check if the customer has communicated the country and city they'd like to go to. If theres enough information, reply 'yes'\\
            if there isnt enough information, reply 'no'. Here is the chat history text between \\{\\}" + "{" + chat_history_text + "}")
# send prompt to llm 
# read the llms response, if the reponse is no, set enough information to false, if yes set enough information to true
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
