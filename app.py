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
    if history is not None:
        history.append((message, bot_message))
    return "", history

# message = "id like to go to Rome , find a nice hotel"
# chat_history = [[]]
# print(multiagent_convo(message, chat_history))


# create chatbot using gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[["", "Where would you like to go?"]])
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(multiagent_convo, [msg, chatbot], [msg, chatbot], queue=False)

demo.launch(share=True,
    server_port=8080)
