import streamlit as st
import json
from snowflake.snowpark import Session
from collections import deque
import base64

# Load credentials
with open("credentials.json", "r") as file:
    credentials = json.load(file)

# Initialize Snowflake session
session = Session.builder.configs(credentials).create()

# Load system prompt
with open("prompt.txt", "r") as file:
    system_prompt = file.read()


def add_custom_css():
    st.markdown("""
        <style>
        .main {
            background: #f9f9f9;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput > div > div > input {
            background: #fff;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stButton > button {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .chat-message {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            background: #fff;
            color:black;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .user-message {
            background: #f0f7f0;
            border-left: 4px solid #4CAF50;
        }
        .wizard-title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #2e7d32;
            margin: 1rem 0;
            padding: 1rem;
        }
        .wizard-subtitle {
            font-size: 1.2em;
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .controls-container {
            background: #fff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        .stSlider > div > div > div {
            background-color: #4CAF50 !important;
        }
        .stSelectbox > div > div {
            background: #fff;
            color:black;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

# Define Custom History class
class CustomHistory:
    def __init__(self, max_len=10):
        self.__history = deque(maxlen=max_len)

    def push(self, user_input, ai_message):
        self.__history.append({"user_input": user_input, "ai_message": ai_message})

    def peek(self, n):
        n = min(n, len(self.__history))
        return list(self.__history)[-n:]

    def __len__(self):
        return len(self.__history)

# Define helper functions with original implementation
def create_prompt(session, myquestion, chat_history, num_chunks=10):
    cmd = """
     with results as
     (SELECT RELATIVE_PATH,
       VECTOR_COSINE_SIMILARITY(GAME_RAG.DATA.DOCS_CHUNKS_TABLE.chunk_vec,
                SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) as similarity,
       chunk
     from docs_chunks_table
     order by similarity desc
     limit ?)
     select chunk, relative_path from results 
     """
    history_str = ""
    for entry in chat_history.peek(num_chunks):
        history_str += f"User: {entry['user_input']}\nAI: {entry['ai_message']}\n"

    df_context = session.sql(cmd, params=[myquestion, num_chunks]).to_pandas()      
    context_lenght = len(df_context) -1
    prompt_context = ""
    for i in range(0, context_lenght):
        prompt_context += df_context._get_value(i, 'CHUNK')
    prompt_context = prompt_context.replace("'", "")
    relative_path = df_context._get_value(0, 'RELATIVE_PATH')
    prompt = f"""
       System Instructions:
       {system_prompt}
       
       Chat History:
       {history_str}
       
       Context: 
       {prompt_context}
       
       Question:  
       {myquestion} 
       
       Answer: 
       """
    return prompt, prompt_context

def chat(session, question, chat_history, model_name, num_chunks=10):
    prompt, prompt_context = create_prompt(session, question, chat_history, num_chunks=num_chunks)
    cmd = """
             SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response
           """
    df_response = session.sql(cmd, params=[model_name, prompt]).collect()
    output = df_response[0].RESPONSE
    chat_history.push(question, output)
    return output, prompt_context

# Initialize chat history
chat_history = CustomHistory(max_len=14)

# Streamlit app
st.set_page_config(
    page_title="Game Wizard",
    layout="centered",
)

# Add custom CSS
add_custom_css()

# Main content
st.markdown("<h1 class='wizard-title'>Game Wizard 🎮</h1>", unsafe_allow_html=True)
st.markdown("<p class='wizard-subtitle'>Your guide to gaming knowledge</p>", unsafe_allow_html=True)

# Chat interface
with st.container():
    # Question input
    question = st.text_input("", placeholder="Ask me anything about games...", key="question_input")
    
    # Controls in a more compact layout
    col1, col2 = st.columns([1, 1])
    with col1:
        num_chunks = st.slider("Knowledge Depth", min_value=1, max_value=20, value=5, 
                             help="Adjust how deep the wizard searches for knowledge")
    with col2:
        model_name = st.selectbox("AI Model", ["mistral-7b"], index=0)

    # Submit button
    if st.button("Ask the Wizard 🎯", key="send_button"):
        if question:
            try:
                with st.spinner("Finding answers..."):
                    answer, context = chat(
                        session=session,
                        question=question,
                        chat_history=chat_history,
                        model_name=model_name,
                        num_chunks=num_chunks
                    )
                st.markdown(f"""
                    <div class='chat-message'>
                        <p>{answer}</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Oops! Something went wrong: {str(e)}")
        else:
            st.warning("Please enter your question first.")

# Simple footer
st.markdown("""
    <div style='text-align: center; margin-top: 30px; opacity: 0.7;'>
        <p>Powered by Gaming Knowledge</p>
    </div>
""", unsafe_allow_html=True)