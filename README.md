## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Current methods for text analysis often struggle with accurately and efficiently extracting contextual information like Named Entities (e.g., person names, locations, organizations) from unstructured text, particularly when dealing with complex linguistic structures. The objective is to implement a robust Transformer-based model (specifically a BERT-based model is used in the code, standing in for the high-performance of a fine-tuned BART/Transformer model) for the NER task, and package it into an intuitive, shareable web application using Gradio. This prototype must demonstrate the capability to correctly identify and categorize entities and merge fragmented word tokens for clean output visualization.

### DESIGN STEPS:

#### STEP 1:
Choose a fine-tuned BART/BERT model for NER and obtain its Hugging Face API endpoint URL.

#### STEP 2:
Load API key securely and create a helper function for communication with the Hugging Face Inference API.

#### STEP 3:
Implement the merge_tokens function to combine sub-word tokens into single, coherent entities.

#### STEP 4:
Define the main ner function to call the API, process the output, and run the token merging logic.

#### STEP 5:
Set up the gr.Interface with a gr.Textbox input and a gr.HighlightedText output for visualization.

#### STEP 6:
Define the title, description, and examples for the Gradio app to enhance user experience.

#### STEP 7:
Deploy the prototype using demo.launch() for real-time testing and evaluation.

### PROGRAM:

```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint
text = "my name is Hezron and i play guitar. i live in chennai with my parents"
get_completion(text, parameters=None, ENDPOINT_URL= API_URL)

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    return {"text": input, "entities": output}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    #Here we introduce a new tag, examples, easy to use examples for your application
                    examples=["my name is Hezron and i play guitar. i live in chennai with my parents"])
demo.launch(share=True, server_port=int(os.environ['PORT3']))

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["my name is Hezron and i play guitar. i live in chennai with my parents"])

demo.launch(share=True, server_port=int(os.environ['PORT4']))

gr.close_all()
```

### OUTPUT:

<img width="1170" height="622" alt="image" src="https://github.com/user-attachments/assets/8e01ab9a-0f78-4193-a165-77805a654483" />

<img width="1166" height="626" alt="image" src="https://github.com/user-attachments/assets/f45f6726-e6a2-41eb-9cd2-0c50639a52c5" />



### RESULT:
Thus, a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation is designed and developed successfully.
