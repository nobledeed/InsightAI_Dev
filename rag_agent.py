from flask import Flask, render_template, request, session, redirect
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()


def chat_rag(user_query):
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]

    user_input = user_query

    def get_context_retriever_chain(vector_store):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

        return retriever_chain

    def get_conversational_rag_chain(retriever_chain):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

        return create_retrieval_chain(retriever_chain, stuff_documents_chain)

    def get_response(user_input):
        retriever_chain = get_context_retriever_chain(vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        response = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })

        return response

    vector_store = FAISS.load_local(folder_path="vector_files", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    response = get_response(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response['answer']})
    session["chat_history"] = chat_history

    return chat_history
