import gradio as gr
from MultiAgent import TravelAssistant
from data_handler import Data_Handler

travel_convo_instance = TravelAssistant()

def multiagent_convo(message, history):
    """
    Function that handles a user query about travel itineraries.

    Parameters:
        message (str): The last user message
        history (list): List of two-element lists of the form [[user_message, bot_message]]

    Returns:
        tuple: A tuple containing an empty string and the updated history
    """

    bot_message = travel_convo_instance.travel_convo(message, history)

    # Check if bot_message is None and handle it
    if bot_message is None:
        bot_message = "[No response generated]"

    # Print the bot message after ensuring it's not None
    print('this is the bot message below ' + bot_message)

    # Append to history if history is not None
    if history is not None:
        history.append((message, bot_message))

    # Return the updated history
    return "", history

# create chatbot using gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[["", "Where would you like to go?"]])
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(multiagent_convo, [msg, chatbot], [msg, chatbot], queue=False)

demo.launch(share=True,
    server_port=8080)
