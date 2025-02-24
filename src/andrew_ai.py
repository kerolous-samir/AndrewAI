from huggingface_hub import login
from google.colab import userdata
hf_token = userdata.get("HF_Token")
login(hf_token, add_to_git_credential=True)
from transformers import pipeline
#import libraries
from transformers import AutoModelForCausalLM , AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
from google.colab import userdata
from diffusers import DiffusionPipeline
import json
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnablePassthrough
# imports

import os
import glob
import gradio as gr
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# imports for langchain

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

from sentence_transformers import SentenceTransformer

class Andrew_Ai:
    hf_token = userdata.get("HF_Token")
    login(hf_token)

    def  __init__(self, image_model = "stabilityai/stable-diffusion-2-1", causal_model = "meta-llama/Llama-3.1-8B-Instruct"):
        self.image_model = image_model
        self.causal_model = causal_model
        self.quant_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.causal_gen = AutoModelForCausalLM.from_pretrained(self.causal_model,quantization_config=self.quant_config,device_map="auto")
        self.img_gen = DiffusionPipeline.from_pretrained(self.image_model,
                                            torch_dtype=torch.float16,use_safetensors=True,
                                            variante="fp16").to("cuda")

        self.history =[]
        self.class_memory =  []

        folders = glob.glob("Knowledgebase/*")
        documents = []
        text_loader_kwargs = {'encoding': 'utf-8'}
        for folder in folders:
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(folder,glob="*.md",loader_cls=TextLoader,loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()
            for file in folder_docs:
                file.metadata["doc_type"] = doc_type
                documents.append(file)
        text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
        db_name = "bhu_db"
        if os.path.exists(db_name):
            Chroma(persist_directory=db_name,embedding_function=embedding_model).delete_collection()
        self.vectorstore = Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory=db_name)


    def generate_response(self,user_prompt:str,system_prompt="your name is andrew and you are IT professional?"):
        self.messages = [{'role': 'system',
          'content': system_prompt}] + self.history + [{'role': 'user', 'content': user_prompt}]
        inputs = self.tokenizer.apply_chat_template(self.messages, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape,device="cuda")
        outputs = self.causal_gen.generate(inputs,attention_mask=attention_mask,
        max_new_tokens=1000)
        response = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        splitter = "assistant\n\n"
        split_text = response.split(splitter)
        ai_response = split_text[-1]
        response_json = {'role': "assistant","content": ai_response}
        self.history.append({'role': 'user', 'content': user_prompt})
        self.history.append({'role': 'assistant', 'content': ai_response})
        del inputs , outputs
        torch.cuda.empty_cache()
        return ai_response

    def generate_img(self,user_prompt:str,system_prompt="""I added to you another model to generate images so please don't say that you not capabile to create image don't even mention it
    and don't mention it at all even don't add note like that:
    (Note: I'm a text-based AI and do not have the capability to display images. However, I can provide vivid descriptions to help you visualize the scene.)
    you will explain the picture that the user asked for in summary and good way like that:

    **Image of a serene mountain landscape at sunset**:
    The image showcases a tranquil mountain range bathed in the warm hues of a setting sun. The sky displays a blend of orange, pink, and purple shades, while calm waters in the foreground reflect the breathtaking colors. Tall, majestic peaks stand in the distance, creating a peaceful and awe-inspiring scene.
    """):

        text_img = self.generate_response(user_prompt,system_prompt)
        img = self.img_gen(text_img).images[0]
        return img , text_img


    def Classification_Model(self,class_prompt):

        img_system_prompt = """You are an AI assistant designed to analyze user prompts and extract content related to image requests, casual conversation, and RAG queries. Your task is to process the user's prompt and output a Python dictionary with exactly the relevant keys: "image", "causal", and "rag".

### Rules:
- "image": If the user requests an image, provide a detailed, expanded description of what they want.
- "causal": If the user engages in casual conversation, preserve the exact text of their message.
- "rag": If the user's prompt is related to HIPAA compliance or the Brain Health USA company, output the relevant query text under this key.
- If the prompt contains only an image request, return only the "image" key.
- If the prompt contains only casual conversation, return only the "causal" key.
- If the prompt contains only a RAG-related query, return only the "rag" key.
- If the prompt contains a combination of these, return all the relevant keys in a clean Python dictionary format with no extra characters (no newlines, no extra whitespace).
- Also, note that memory is added to check for image requests but not for casual conversation.

### Examples:

#### 1️⃣ When the user only asks for an image
User Prompt: "Draw me a castle in the mountains."
Correct Output:
{"image": "A grand medieval castle situated high in the mountains, surrounded by mist and dense pine forests, with a river flowing in the valley below."}

#### 2️⃣ When the user only engages in casual conversation
User Prompt: "Hey, how are you doing today?"
Correct Output:
{"causal": "Hey, how are you doing today?"}

#### 3️⃣ When the user asks for both an image and casual conversation
User Prompt: "Draw me a beach at sunset, and by the way, how’s your day?"
Correct Output:
{"image": "A peaceful beach at sunset with golden sands, gentle waves reflecting vibrant oranges and pinks, and palm trees swaying in the breeze.", "causal": "how’s your day?"}

#### 4️⃣ When the user's query is related to HIPAA compliance or Brain Health USA company
User Prompt: "Can you tell me the latest HIPAA compliance regulations for healthcare providers?"
Correct Output:
{"rag": "Can you tell me the latest HIPAA compliance regulations for healthcare providers?"}

User Prompt: "I need information about Brain Health USA company policies."
Correct Output:
{"rag": "I need information about Brain Health USA company policies."}

#### 5️⃣ When the user's prompt includes image, casual conversation, and RAG together
User Prompt: "Draw me a futuristic cityscape, how's your day, and can you provide the latest HIPAA compliance guidelines for healthcare providers?"
Correct Output:
{"image": "A futuristic cityscape with towering skyscrapers made of glass and neon lights, flying vehicles zooming past, and an urban environment buzzing with energy.", "causal": "how's your day?", "rag": "can you provide the latest HIPAA compliance guidelines for healthcare providers?"}
"""


        self.class_message = [{'role': 'system',
          'content': img_system_prompt}] + self.class_memory + [{'role': 'user', 'content': class_prompt}]
        inputs = self.tokenizer.apply_chat_template(self.class_message, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape,device="cuda")
        outputs = self.causal_gen.generate(inputs,attention_mask=attention_mask,
        max_new_tokens=1000)
        response = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        splitter = "assistant\n\n"
        split_text = response.split(splitter)
        list = split_text[-1]
        if "image" in json.loads(list).keys():
          self.class_memory.append({'role': 'user', 'content': json.loads(list)['image']})
          self.class_memory.append({'role': 'assistant', 'content': json.loads(list)['image']})
        torch.cuda.empty_cache()
        return list

    def RAG(self,user_prompt):
      retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
      memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
      qa_chain = ConversationalRetrievalChain.from_llm(
      self.causal_gen,
      retriever,
      memory=memory
      )
      result = qa_chain({"question": user_prompt})
      self.history.append({'role': 'user', 'content': user_prompt})
      self.history.append({'role': 'assistant', 'content': result['answer']})

      return result['answer']



    def chat(self,user_prompt):
      classy = self.Classification_Model(user_prompt)
      classy = json.loads(classy)
      for i in range(len(list(classy.keys()))):
        if "causal" in list(classy.keys())[i]:
          chat = self.generate_response(user_prompt=classy['causal'])
          yield display(chat)
        if "image" in list(classy.keys())[i]:
          image , tex = self.generate_img(user_prompt=classy['image'])
          yield display(tex)
          yield display(image)

        if "rag" in list(classy.keys())[i]:
          rag = self.RAG(user_prompt=classy['rag'])
          yield display(rag)
        




