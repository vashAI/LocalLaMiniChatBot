# Importing necessary modules
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
import torch
import warnings

import streamlit as st
from time import  sleep
from langchain.memory import ConversationBufferWindowMemory


warnings.filterwarnings("ignore", category=UserWarning)

# Setting up avatars for chat messages
human = './human.png'  
robot = './robot.png'

# Initializing the Lamini model and pipeline
checkpoint = "./model/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                    device_map='auto',
                                                    torch_dtype=torch.float32)

llm = HuggingFacePipeline.from_model_id(model_id=checkpoint,
                                        task = 'text2text-generation',
                                        model_kwargs={"temperature":0.60,"min_length":35, "max_length":500, "repetition_penalty": 5.0})
from langchain import PromptTemplate, LLMChain
template = """{text}"""
prompt = PromptTemplate(template=template, input_variables=["text"])
chat = LLMChain(prompt=prompt, llm=llm)

# Displaying chatbot interface title and description
st.title("Offline LaMini ChatBot (100% Offline 100% Privacy)")
st.subheader("ChatBot based on a local LaMini 248 Parameter LLM Model")

# Checking and initializing chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Displaying previous chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"],avatar=human):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"],avatar=robot):
            st.markdown(message["content"])

# Accepting user input
if myprompt := st.chat_input("How can I help you?"):
    
    # Adding user message to chat history
    st.session_state.messages.append({"role": "user", "content": myprompt})
  
    # Displaying user message in chat message container
    with st.chat_message("user", avatar=human):
        st.markdown(myprompt)
        usertext = f"user: {myprompt}"
    
    # Generating and displaying assistant response in chat message container
    with st.chat_message("assistant", avatar=robot):
        message_placeholder = st.empty()
        full_response = ""
        res  =  chat.run(myprompt)
        response = res.split(" ")
        for r in response:
            full_response = full_response + r + " "
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
        asstext = f"assistant: {full_response}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
