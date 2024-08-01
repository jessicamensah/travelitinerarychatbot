import gradio as gr # type: ignore
from langchain_community.llms import Ollama
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
        # Construct a concise prompt
        prompt = (
            "Travel itinerary agent. Based on the chat history, provide recommendations or ask for more info. "
            "Chat history: " + history + "\n" + message
        )
        # Limit the prompt to a specific number of words
        prompt = limit_words(prompt, 100)  # Adjust the word limit as needed
        llm = Ollama(model="phi3:3.8b")
        response = llm.invoke(prompt)
        
        # Post-processing check for hallucinations
        if "fabricated" in response or "assumption" in response:
            return "The response seems to contain hallucinations. Please provide more details."
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"
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
