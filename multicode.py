import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import pickle
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback

st.markdown("""
<style>
    .userbox {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #0a0a0a;
        color: white;
    }

    .responsebox {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #c7e0b9;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

def initialize_database():
    conn = sqlite3.connect('temp3.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(chat_bot)")
    columns = [info[1] for info in cursor.fetchall()]
    if "user_input" in columns and "chatbot_id" not in columns:
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS new_chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

        cursor.execute('''
            INSERT INTO new_chat_bot (user_input, chatbot_response)
            SELECT user_input, chatbot_response FROM chat_bot
        ''')
        conn.commit()

        cursor.execute("DROP TABLE chat_bot")
        cursor.execute("ALTER TABLE new_chat_bot RENAME TO chat_bot")
        conn.commit()

    else:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_bot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chatbot_id TEXT,
                user_input TEXT,
                chatbot_response TEXT
            )
        ''')
        conn.commit()

    return conn, cursor


def clear_history(conn, cursor, chatbot_id):
    cursor.execute('DELETE FROM chat_bot WHERE chatbot_id = ?', (chatbot_id,))
    conn.commit()

def create_vector_store_from_pdf(pdf_file):
    loader = PyPDFLoader(file_path=pdf_file)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    texts = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(texts, embeddings)
    return vectordb

def save_vector_store(vectordb, filename="faiss_store.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(vectordb, f)

def load_vector_store(filename="faiss_store.pkl"):
    with open(filename, "rb") as f:
        vectordb = pickle.load(f)
    return vectordb


prompt_template = """
    given the following question. generate based on the context only. try to provide 
    as much as text as possible from "response" context without making a mistake.
    If the answer is not found in the context, kindly state "I don't Know." Don't 
    try to make up an answer.

    CONTEXT: {context}
    QUESTION: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def aibot(prompt, chatbot_id, chat_bot, conn, cursor):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a very friendly and creative AI"),
        ("user", prompt)
    ])
    
    response = chat_bot(chat_prompt.format_messages(prompt=prompt), temperature=0.65, max_tokens=300)
    response=response.content
    response_text = response if isinstance(response, str) else str(response)

    cursor.execute("INSERT INTO chat_bot (chatbot_id, user_input, chatbot_response) VALUES (?, ?, ?)",
                   (chatbot_id, prompt, response_text))
    conn.commit()
    
    return response_text



def main():
    st.title("Multi-Chatbot Manager 🤖")

    conn, cursor = initialize_database()

    if 'chatbots' not in st.session_state:
        st.session_state.chatbots = {}

    new_chatbot_id = st.sidebar.text_input("Enter new chatbot ID:")
    if st.sidebar.button("Add Chatbot"):
        if new_chatbot_id not in st.session_state.chatbots:
            st.session_state.chatbots[new_chatbot_id] = {}

    selected_chatbot_id = st.sidebar.selectbox("Select a chatbot:", list(st.session_state.chatbots.keys()))

    if st.sidebar.button("Delete Selected Chatbot"):
        if selected_chatbot_id and selected_chatbot_id in st.session_state.chatbots:
            cursor.execute("DELETE FROM chat_bot WHERE chatbot_id = ?", (selected_chatbot_id,))
            conn.commit()

            del st.session_state.chatbots[selected_chatbot_id]
            st.experimental_rerun() 

    if selected_chatbot_id:
        chatbot_controls(selected_chatbot_id, conn, cursor)

    conn.close()

def chatbot_controls(chatbot_id, conn, cursor):

    st.sidebar.markdown(f"## Chatbot: {chatbot_id}")

    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

    if chatbot_id not in st.session_state.chatbots:
        st.session_state.chatbots[chatbot_id] = {"model": None, "temperature": 0.7, "max_tokens": 500, "vectordb": None}

    chatbot = st.session_state.chatbots[chatbot_id]

    chatbot['model'] = st.sidebar.selectbox(f"Select model for {chatbot_id}", ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301", "gpt-4-0314"], key=f"model_{chatbot_id}")
    chatbot['temperature'] = st.sidebar.slider(f"Select temperature for {chatbot_id}", min_value=0.20, max_value=1.00, value=0.40, step=0.20, key=f"temperature_{chatbot_id}")
    chatbot['max_tokens'] = st.sidebar.slider(f"Select max_tokens for {chatbot_id}", min_value=200, max_value=1200, value=400, step=200, key=f"max_tokens_{chatbot_id}")

    if 'file_name' not in chatbot:
        chatbot['file_name'] = None

    uploaded_file = st.sidebar.file_uploader(f"Upload a PDF document for {chatbot_id}", type="pdf", key=f"file_uploader_{chatbot_id}")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        chatbot['vectordb'] = create_vector_store_from_pdf(tmp_file_path)
        save_vector_store(chatbot['vectordb'])
        chatbot['file_name'] = file_name

    if chatbot['file_name']:
        st.sidebar.markdown(f"Uploaded file: `{chatbot['file_name']}`")

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=chatbot['model'], temperature=chatbot['temperature'], max_tokens=chatbot['max_tokens'])

    user_input = st.sidebar.text_input(f"Ask a question to {chatbot_id}:", key=f"user_input_{chatbot_id}")

    if st.sidebar.button(f"Submit to {chatbot_id}", key=f"submit_{chatbot_id}"):

        if chatbot.get('vectordb'):
            vectordb = chatbot['vectordb']

            doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

            chain = ConversationalRetrievalChain(
                retriever=vectordb.as_retriever(), 
                combine_docs_chain=doc_chain, 
                question_generator=LLMChain(
                    llm=llm, 
                    prompt=prompt, 
                    verbose=True, 
                    memory=ConversationBufferWindowMemory(k=4),
                ),
                return_source_documents=True
            )

            with get_openai_callback() as cb:
                response = chain({
                    "chat_history": [],  
                    "question": user_input  
                })
            st.sidebar.write(cb)

            response = response["answer"]

            cursor.execute("INSERT INTO chat_bot (chatbot_id, user_input, chatbot_response) VALUES (?, ?, ?)", (chatbot_id, user_input, response))
            conn.commit()

            st.markdown(f"<div class='userbox'>Chatbot {chatbot_id}: {response}</div>", unsafe_allow_html=True)

        else:
            chat_bot = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            response = aibot(user_input, chatbot_id, chat_bot, conn, cursor)
            st.markdown(f"<div class='userbox'>Chatbot {chatbot_id}: {response}</div>", unsafe_allow_html=True)

    st.markdown(f"### Chat History for {chatbot_id}")
    col1, col2 = st.columns([3, 1])     

    col1.header("Chat History 📜")
    if col2.button("Clear History 🧹"):
        clear_history(conn, cursor, chatbot_id)

    history = cursor.execute("SELECT user_input, chatbot_response FROM chat_bot WHERE chatbot_id = ?", (chatbot_id,)).fetchall()
    for user_input, chatbot_response in history[::-1]:
        st.markdown(f"<div class='userbox'>User: {user_input}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='responsebox'>Chatbot {chatbot_id}: {chatbot_response}</div>", unsafe_allow_html=True)
            

if __name__ == "__main__":
    main()
