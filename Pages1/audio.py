import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'mic_output' not in st.session_state:
    st.session_state['mic_output'] = None
if 'transcribed_text' not in st.session_state:
    st.session_state['transcribed_text'] = ""

# Streamlit UI
st.title("Speech-to-Text Conversion")

# Function to display the transcribed text and store it in a variable
def stt_callback():
    if st.session_state.stt_output:
        transcribed_text = st.session_state.stt_output
        st.write("Transcribed Text:", transcribed_text)
        st.session_state.transcribed_text = transcribed_text

# Speech-to-text widget
speech_to_text(key='stt', callback=stt_callback)

# Audio recorder widget
audio = mic_recorder()

# Display the recorded audio
if st.session_state.mic_output:
    st.audio(st.session_state.mic_output['audio'], format='audio/wav')

# Display the transcribed text stored in the variable
if st.session_state.transcribed_text:
    st.write("Stored Transcribed Text:", st.session_state.transcribed_text)

# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the queries."),
    ("user", "Question: {question}")
])

llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Initialize conversation
@st.cache(allow_output_mutation=True)
def get_conversation():
    return []

conversation = get_conversation()

# Submit button action
if st.button("Submit"):
    if st.session_state.transcribed_text.strip().lower() == "quit":
        conversation.clear()  # Clear conversation if user wants to quit
    else:
        conversation.append(("user", st.session_state.transcribed_text))
        chain = prompt.from_messages(conversation) | llm | output_parser
        output = chain.invoke({'question': st.session_state.transcribed_text})
        conversation.append(("system", output))

# Display conversation
for msg_type, msg_content in conversation:
    if msg_type == "user":
        st.write(f"User: {msg_content}")
    else:
        st.write(f"System: {[msg_content]}")
