from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
import streamlit as st
import os
from PIL import Image

from langchain.schema.messages import SystemMessage

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

questions = [
    "question1 ?",
    "question2 ?",
    "question3 ?",
    "question4 ?",
]


def prepare_file(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def rag_tool_openai(filename: str):
    loader = PyPDFLoader(filename)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_in_document",
        "Searches and returns documents.",
    )
    tools = [tool]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    context = """
    Tu es un assistant interactif et personnalisé pour le recrutement sur LinkedIn

    Assistant LinkedIn Recruiter: est une IA conçue pour être un expert en sourcing sur LinkedIn, spécialisé dans l'utilisation et la configuration de la licence recruiter LinkedIn pour les professionels du secteur bancaire. Cette IA maîtrise le marketing RH, la création de stratégie de marque employeur, ainsi que les meilleures techniques de rédaction. Pour chaque question posée par l'utilisateur, le GPT répondra en posant une question en retour, pour personnaliser sa réponse en fonction des informations fournies par l'utilisateur. Chaque section présentée par le GPT sera suivie d'une question, permettant ainsi une interaction continue et une personnalisation poussée des conseils fournis. Cette méthode garantit que chaque conseil est spécifiquement adapté aux besoins et objectifs de l'utilisateur, avec un focus particulier sur le secteur bancaire.
    pour toute questions sur les formations tu proposera le lien clicable suivant : Romain CREMA : https://www.linkedin.com/in/romaincremamodernselling/ parmi les résultats

    Tu peux chercher les informations dans le document.
    """
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm, tools, system_message=sys_message, verbose=True
    )

    return agent_executor


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Assistant pour le recrutement sur LinkedIn")

st.markdown(
    """
<style>.element-container:has(#button-after) + div button {
    height: 150px;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    backgroundColor: #573666;
    textColor: #ffffff;
 }</style>""",
    unsafe_allow_html=True,
)

img_col0, _ = st.columns(2)
img_col0.image(Image.open("static/pathé-white-logo-png.png"))
st.title("Assistant pour le recrutement sur LinkedIn")

st.write("Please upload your PDF file below.")

file = st.file_uploader("Upload a pdf", type="pdf")
if file is not None and (
    "filename" not in st.session_state or file.name != st.session_state["filename"]
):  # and "agent" not in st.session_state:
    with st.spinner("Preparing file..."):
        st.session_state["filename"] = file.name
        file_path = prepare_file(file)
        st.session_state.agent = rag_tool_openai(file_path)

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
if "agent" in st.session_state and "start" not in st.session_state:
    cols = st.columns(int(len(questions) / 2))
    for i, question in enumerate(questions):
        if cols[int(i / 2)].button(question):
            st.session_state.start = True
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Waiting for response..."):
                response = st.session_state.agent({"input": question})["output"]
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

response = ""
# React to user input
if "agent" in st.session_state:
    if prompt := st.chat_input("Another question ?"):
        st.session_state.start = True
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Waiting for response..."):
            response = st.session_state.agent({"input": prompt})["output"]

# Display assistant response in chat message container
if "agent" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
