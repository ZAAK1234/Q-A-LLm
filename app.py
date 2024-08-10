
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from streamlit.elements import text                    
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os
import google.generativeai as genai
import streamlit as st
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub
from style import css, bot_template, user_template
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_file_text(pdf_docs):
    text = ""
    for file in pdf_docs:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for paragraph in doc.paragraphs:
                text += paragraph.text
        elif file.type == "text/plain":
            text += file.read().decode('utf-8')
    return text

def get_text_chunks(text):
  text_splitter=CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks=text_splitter.split_text(text)
  return chunks

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error in get_vector_store: {e}")
        return None
    
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3,top_p=0.85)
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory
       
   )
    return conversation_chain

def handle_user_input(user_question):
   response=st.session_state.conversation({"question":user_question})
   st.session_state.chat_history=response['chat_history']

   for i, message in enumerate(st.session_state.chat_history):
      if i % 2 == 0:
        st.write(user_template(f'Question : {message.content}'),unsafe_allow_html=True)
      else:
        st.write(bot_template(message.content),unsafe_allow_html=True)
   
  
  


def main():
  load_dotenv()
  os.getenv("GOOGLE_API_KEY")
  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
  st.set_page_config(page_title="Chatbot", page_icon=":robot_face:")
  st.write(css, unsafe_allow_html=True)
  if "conversation" not in st.session_state:
    st.session_state.conversation=None

  if "chat_history" not in st.session_state:
    st.session_state.chat_history=None
  st.header("Chatbot")

  user_question=st.text_input("Welcome to the Chatbot!")
  if user_question:
    handle_user_input(user_question)
  
  with st.sidebar:
    st.subheader("Data")
    pdf_docs=st.file_uploader("Upload PDF Documents",accept_multiple_files=True)
    if st.button("Upload"):
      with st.spinner("Processing..."):
        #get pdf text
        raw_text = get_file_text(pdf_docs)
        
          #get the text chunks
        text_chunks = get_text_chunks(raw_text)
        

          #get the vector
        vectorstore=get_vector_store(text_chunks)

        #create conversation chat
        st.session_state.conversation=get_conversation_chain(vectorstore)
  



if __name__ == "__main__":
  main()
      
