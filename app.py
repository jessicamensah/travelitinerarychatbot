import gradio as gr
from MultiAgent import travel_convo

travel_convo_instance = travel_convo()

def multiagent_convo(message, history):
    """
    Function that handles a user query about travel itineraries.

    Parameters:
        message (str): The last user message
        history (list): List of two-element lists of the form [[user_message, bot_message]]

    Returns:
        tuple: A tuple containing an empty string and the updated history
    """
    bot_message = travel_convo_instance.find_travel_chat(message, history)
    history.append((message, bot_message))
    return "", history


# Create chatbot using Gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[["", "Hi! I'm your Travel Itinerary Bot, where would you like to go?"]])
    msg = gr.Textbox(show_label=False, placeholder="Enter your message here...")
    clear = gr.Button("Clear")

    def clear_chat():
        return gr.update(value=[]), gr.update(value="")

    msg.submit(multiagent_convo, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(clear_chat, [], [chatbot, msg])

demo.launch(
server_port=8080)