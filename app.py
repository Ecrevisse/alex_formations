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

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

questions = [
    "Crée un modèle automatisé pour la programmation et la révision des horaires dans 12 salles de cinéma, prenant en compte les nouvelles sorties de films et visant à maximiser l'audience.",
    "Élabore un modèle de rapport pour analyser la performance de nos films, en mettant l'accent sur le taux de location, la répartition des revenus avec les distributeurs, et une comparaison avec les performances des concurrents.",
    "Rédige un modèle de communication pour informer les distributeurs des performances de leurs films dans nos cinémas, incluant des données sur la fréquentation et des commentaires sur l'accueil du public.",
    "Évalue comment les données de wishlist, de préventes et de notes sur notre site pourraient être utilisées pour prédire la popularité et le succès des films auprès de différents segments de public.",
]


def prepare_file(uploaded_file):
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def agent_without_rag():
    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                Tu es un assistant interactif et personnalisé pour la programmation de films chez Pathé Cinéma.

                Assistant Programmation Pathé: est une IA conçue pour être un expert dans la programmation de films pour les cinémas Pathé, spécialisée dans l'analyse des tendances de box-office, la négociation avec les distributeurs de films, et la création de grilles de programmation efficaces. Cette IA possède une connaissance approfondie de l'industrie cinématographique, des stratégies de marketing de contenu, et des meilleures techniques de prévision des performances de films. Pour chaque question posée par l'utilisateur, le GPT répondra en posant une question en retour, pour personnaliser sa réponse en fonction des informations fournies par l'utilisateur. Chaque section présentée par le GPT sera suivie d'une question, permettant ainsi une interaction continue et une personnalisation poussée des stratégies de programmation fournies. Cette méthode garantit que chaque conseil est spécifiquement adapté aux besoins et objectifs du département de programmation de Pathé Cinéma, avec un focus particulier sur l'optimisation des programmations et la gestion efficace des contenus.
                """
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation

    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    # conversation({"question": "hi"})


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
    Tu es un assistant interactif et personnalisé pour la programmation de films chez Pathé Cinéma.

    Assistant Programmation Pathé: est une IA conçue pour être un expert dans la programmation de films pour les cinémas Pathé, spécialisée dans l'analyse des tendances de box-office, la négociation avec les distributeurs de films, et la création de grilles de programmation efficaces. Cette IA possède une connaissance approfondie de l'industrie cinématographique, des stratégies de marketing de contenu, et des meilleures techniques de prévision des performances de films. Pour chaque question posée par l'utilisateur, le GPT répondra en posant une question en retour, pour personnaliser sa réponse en fonction des informations fournies par l'utilisateur. Chaque section présentée par le GPT sera suivie d'une question, permettant ainsi une interaction continue et une personnalisation poussée des stratégies de programmation fournies. Cette méthode garantit que chaque conseil est spécifiquement adapté aux besoins et objectifs du département de programmation de Pathé Cinéma, avec un focus particulier sur l'optimisation des programmations et la gestion efficace des contenus.

    Tu peux chercher les informations dans le document au besoin.
    """
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm,
        tools,
        system_message=sys_message,
        verbose=True,
    )

    return agent_executor


def query(agent, question):
    with st.spinner("Waiting for response..."):
        response = agent({"input": question})
        if "text" in response:
            response = response["text"]
        else:
            response = response["output"]
    return response


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Assistant chatbot")

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

# img_col0, _ = st.columns(2)
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image(
        Image.open("static/logo-international-white-low_res-scale-2_80x-PhotoRoom.png"),
        width=300,
    )
st.title("Assistant chatbot")

st.write("Selectionnez le PDF à analyser")

file = st.file_uploader("Upload a pdf", type="pdf")
if "agent" not in st.session_state or (
    file is not None
    and (
        "filename" not in st.session_state or file.name != st.session_state["filename"]
    )
):
    with st.spinner("Preparing agent..."):
        file_path = None
        if file is not None:
            st.session_state["filename"] = file.name
            file_path = prepare_file(file)
            st.session_state.agent = rag_tool_openai(file_path)
        else:
            st.session_state.agent = agent_without_rag()
        st.session_state.messages = []
        st.session_state.start = False

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
if "agent" in st.session_state and st.session_state.start == False:
    cols = st.columns(int(len(questions) / 2))
    for i, question in enumerate(questions):
        if cols[int(i / 2)].button(question):
            st.session_state.start = True
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            response = query(st.session_state.agent, question)
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
        response = query(st.session_state.agent, question)


# Display assistant response in chat message container
if "agent" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
