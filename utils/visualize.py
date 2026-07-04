import gradio as gr
import json
import requests
from io import BytesIO
from PIL import Image

def load_json_file(json_file):
    """
    When a JSON file is uploaded, load the data and update the dropdown choices.
    The data is stored in a hidden state.
    """
    if json_file is None:
        return gr.update(choices=[]), None
    try:
        # Depending on your Gradio version, json_file may be a file path or file-like object.
        if hasattr(json_file, "name"):
            with open(json_file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.load(json_file)
    except Exception as e:
        print("Error reading JSON:", e)
        return gr.update(choices=[]), None

    # Create dropdown choices based on the image_id from each dict.
    choices = [str(item["image_id"]) for item in data]
    default = choices[0] if choices else None
    return gr.update(choices=choices, value=default), data

def update_display(image_id, json_data):
    """
    Given an image_id and the loaded JSON data, download the image and prepare
    a conversation history list for display in a Chatbot component.
    """
    if json_data is None or image_id is None:
        return None, []
    
    # Find the item with the matching image_id.
    selected_item = next((item for item in json_data if str(item["image_id"]) == image_id), None)
    if selected_item is None:
        return None, []
    
    try:
        response = requests.get(selected_item["url"])
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        print("Error downloading image:", e)
        image = None

    # Build a conversation list as tuples (prompt, response) for the Chatbot.
    chat_history = []
    for conv in selected_item.get("conversations", []):
        chat_history.append((conv.get("prompt", ""), conv.get("response", "")))
    
    return image, chat_history

# Build the Gradio interface.
with gr.Blocks() as demo:
    gr.Markdown("## MLLM Visualizer")
    with gr.Row():
        with gr.Column(scale=1):
            json_file = gr.File(label="Upload JSON file", file_types=[".json"])
            dropdown = gr.Dropdown(label="Select Image ID", choices=[])
        with gr.Column(scale=2):
            image_out = gr.Image(label="Image")
            chatbot = gr.Chatbot(label="Conversation")
    
    # A hidden state component to store the loaded JSON data.
    json_state = gr.State()

    # When a file is uploaded, load the JSON data and update the dropdown.
    json_file.change(load_json_file, inputs=json_file, outputs=[dropdown, json_state])
    # When the dropdown selection changes, update the image and conversation.
    dropdown.change(update_display, inputs=[dropdown, json_state], outputs=[image_out, chatbot])

# Launch with a public share link (if desired).
demo.launch(share=True)
