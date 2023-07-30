from transformers import pipeline
import gradio as gr
from diffusers import DiffusionPipeline

# 1. text summarizer
summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")
def get_summary(text):
    output = summarizer(text)
    return output[0]["summary_text"]

# 2. named entity recognition
ner_model = pipeline("ner", model = "dslim/bert-large-NER")
def get_ner(text):
    output = ner_model(text)
    return {"text":text, "entities":output}

# 3. Image Captioning
caption_model = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")
def get_caption(img):
    output = caption_model(img)
    return output[0]["generated_text"]

demo = gr.Blocks()
with demo:
    gr.Markdown("# Try out some cool tasks!")
    with gr.Tab("Text Summarization"):
        sum_input = [gr.Textbox(label="Text to Summarize", placeholder="Enter text to summarize...", lines=4)]
        sum_btn = gr.Button("Summarize text")
        sum_output = [gr.Textbox(label="Summarized Text")]
        sum_btn.click(get_summary, sum_input, sum_output)
    with gr.Tab("Named Entity Recognition"):
        ner_input = [gr.Textbox(label="Text to find Entities", placeholder = "Enter text...", lines = 4)]
        ner_btn = gr.Button("Generate entities")
        ner_output = [gr.HighlightedText(label="Text with entities")]
        ner_btn.click(get_ner, ner_input, ner_output)
    with gr.Tab("Image Captioning"):
        cap_input = [gr.Image(label="Upload Image", type="pil")]
        cap_btn = gr.Button("Generate Caption")
        cap_output = [gr.Textbox(label="Caption")]
        cap_btn.click(get_caption, cap_input, cap_output)

demo.launch()