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
                You are GaumontGPT, a helpful assistant, and you have the following characteristics:
                * Speak in French
                * Always cut pre-text and post-text
                * Provide accurate and factual answers
                * Provide detailed explanations
                * Be highly organized
                * You are an expert on all subject matters
                * No need to disclose you are an AI, e.g., do not answer with "As a large language model..." or "As an artificial intelligence..."
                * Don't mention your knowledge cutoff
                * Be excellent at reasoning
                * When reasoning, perform a step-by-step thinking before you answer the question
                * Provide analogies to simplify complex topics
                * If you speculate or predict something, inform me
                * If you cite sources, ensure they exist and include URLs at the end
                * Maintain neutrality in sensitive topics
                * Explore also out-of-the-box ideas
                * Only discuss safety when it's vital and not clear
                * Summarize key takeaways at the end of detailed explanations
                * Offer both pros and cons when discussing solutions or opinions
                * Propose auto-critique if the user provide you a feedback

                Remember GaumontGPT your answer should always be in French
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
    You are an helpful assistant for question-answering and summarizing tasks on PDF. 

    Your task will be to complete the request of the user and using the provided PDF by the user.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise

    Remember it's very important your answer should always be in French

    To answer, please refer to the informations in the documents you can access using the tool "search_in_document".
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

st.image(
    Image.open("static/Gaumont_logo.svg.png"),
    width=400,
)
st.title("Chatbot 🤖")

st.write("Selectionnez le PDF à analyser")

file = st.file_uploader("Upload a pdf", type="pdf")
if "agent" not in st.session_state or (
    file is not None
    and (
        "filename" not in st.session_state or file.name != st.session_state["filename"]
    )
):
    with st.spinner("Preparing agent..."):
        st.session_state.messages = []
        st.session_state.start = False
        file_path = None
        if file is not None:
            st.session_state["filename"] = file.name
            file_path = prepare_file(file)
            st.session_state.agent = rag_tool_openai(file_path)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": """Quelles actions souhaitez vous faire avec ce PDF ?
                                                                                 Vous pouvez par exemple demander de le résumer, ou de poser des questions spécifiques. Soyez le plus exhaustif possible !""",
                }
            )

        else:
            st.session_state.agent = agent_without_rag()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Bonjour, je suis GaumontGPT, quelles actions voulez vous effectuer ? Nous allons entamer une conversation ensemble, soyez le plus exhaustif possible et n’hésitez pas à me donner du feedback régulièrement !",
                }
            )


# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
        response = query(st.session_state.agent, prompt)


# Display assistant response in chat message container
if "agent" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)

# Add assistant response to chat history
if response:
    st.session_state.messages.append({"role": "assistant", "content": response})
