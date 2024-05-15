# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


# from langchain_community.llms import Ollama
# import streamlit as st
# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# input_text = st.text_input('Search for the topic you want')

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant please respond to the queries"),
#         ("user", "Question:{question}")
#     ]
# )

# # StreamLit Framework

# st.title('LangChain Demo with LLAMA')

# # OpenAI LLM
# llm =  Ollama(model="llama2")
# output_parser = StrOutputParser()  # Getting the output
# chain = prompt | llm | output_parser  # Combine all the above 3

# if input_text:
#     st.write(chain.invoke({'question': input_text}))
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langserve import add_routes
# import uvicorn


# load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Initialize LangChain components
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. Please respond to the queries."),
#     ("user", "Question: {question}")
# ])

# llm = Ollama(model="llama2")
# output_parser = StrOutputParser()

# # Streamlit Framework
# st.title('LangChain Demo with LLAMA')
# conversation = []

# input_text = st.text_input('Ask a question or type "quit" to exit the conversation:', key="input_text")

# if st.button("Submit"):
#     if input_text.strip().lower() == "quit":
#         conversation.clear()  # Clear conversation if user wants to quit
#     else:
#         conversation.append(("user", input_text))
#         chain = prompt.from_messages(conversation) | llm | output_parser
#         output = chain.invoke({'question': input_text})
#         conversation.append(("system", output))

# for msg_type, msg_content in conversation:
#     if msg_type == "user":
#         st.write(f"User: {msg_content}")
#     else:
#         st.write(f"System: {msg_content}")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the queries."),
    ("user", "Question: {question}")
])

llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Streamlit Framework
st.title('LangChain Demo with LLAMA')

@st.cache(allow_output_mutation=True)
def get_conversation():
    return []

conversation = get_conversation()

input_text = st.text_input('Ask a question or type "quit" to exit the conversation:', key="input_text")

if st.button("Submit"):
    if input_text.strip().lower() == "quit":
        conversation.clear()  # Clear conversation if user wants to quit
    else:
        conversation.append(("user", input_text))
        chain = prompt.from_messages(conversation) | llm | output_parser
        output = chain.invoke({'question': input_text})
        conversation.append(("system", output))

for msg_type, msg_content in conversation:
    if msg_type == "user":
        st.write(f"User: {msg_content}")
    else:
        st.write(f"System: {msg_content}")


