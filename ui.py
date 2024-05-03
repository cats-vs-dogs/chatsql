import streamlit as st
from chatbot import Chatbot
from langchain.memory import ConversationBufferWindowMemory

# add a heading for your app.
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 1.3rem;
    }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.subheader(r"$\texttt{\huge ChatSQL}$")

# st.header("Chat with your data")
# st.markdown('---')

# Initialize the memory
# This is needed for both the memory and the prompt
memory_key = "history"

if "memory" not in st.session_state.keys():
    # st.session_state.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key=memory_key, return_messages=True)

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?!"}
    ]


# Prompt for user input and display message history
if prompt := st.chat_input("Ask something"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = Chatbot().agent_executor({"input": prompt})
            st.write(response["output"])           
            message = {"role": "assistant", "content": response["output"]}
            st.session_state.messages.append(message) # Add response to message history
