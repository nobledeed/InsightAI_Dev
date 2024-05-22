import sqlite3

from flask import Flask, render_template, request, session, redirect

from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv

from langchain_community.agent_toolkits import create_sql_agent

import os.path
import shutil
import research_agent
import rag_agent

load_dotenv()

app = Flask(__name__)


def get_db_connection():
    conn = sqlite3.connect("/home/nobledeed/PycharmProjects/InsightAI_Dev/sqlite.db")
    return conn


app.secret_key = os.environ.get('FLASK_KEY')
#app.config["RAG_files"] = "/home/nobledeed/PycharmProjects/InsightAI/RAG_files"


@app.route("/end_chat", methods=["POST"])
def end_chat():
    session.clear()
    conn = get_db_connection()
    conn.execute('DELETE FROM message_store')
    conn.commit()
    conn.close()
    return render_template("_chat_form.html")


@app.route("/end_rag_chat", methods=["POST"])
def end_chat_rag():
    session.clear()
    file = os.listdir("./static/rag/")
    pdf_file = file[0]

    return render_template("_rag_chat_form.html", pdf_file=pdf_file)


@app.route("/delete_rag_files", methods=["POST"])
def delete_rag():
    session.clear()
    file = os.listdir("./static/rag/")
    os.remove(os.path.join("./static/rag/", file[0]))
    folder_path = "/home/nobledeed/PycharmProjects/InsightAI_Dev/vector_files/"
    shutil.rmtree(folder_path)
    return render_template("_rag_upload.html")


@app.route("/")
def landing_page():
    body_type = "body"
    return render_template("index.html", body_type=body_type,page = "home")


@app.route("/assistant_chat")
def assistant_page():
    session.clear()
    body_type = "body class='subpage'"
    return render_template("assistant_chat.html", body_type=body_type, page ="assistant")


@app.route("/sql_chat")
def sql_page():
    session.clear()
    body_type = "body class='subpage'"
    return render_template("sql_chat.html", body_type=body_type, page = "sql")


@app.route("/elements")
def elements_page():
    return render_template("elements.html")


@app.route("/chat")
def chat_page():
    session.clear()
    return render_template("_chat_form.html")


# @app.route("/rag_chat")
# def rag_chat_page():
#     session.clear()
#
#     return render_template("_rag_chat_form.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_query = request.form["user_input"]

    if user_query is not None and user_query != "":
        chat_response = research_agent.chat_assistant(user_query)

        return render_template("_chat.html", chat_history=chat_response.messages)


@app.route("/get_sql_response", methods=["POST"])
def get_sql_response():
    user_query = request.form["user_input"]
    db = SQLDatabase.from_uri("sqlite:///online_store.db")

    llm = ChatOpenAI(temperature=0)

    agent_executor = create_sql_agent(
        llm, db=db, agent_type="openai-tools", verbose=True
    )

    if user_query is not None and user_query != "":
        response = agent_executor.invoke(user_query)

        return render_template("_sql_chat.html", result=response["output"])


@app.route('/home_rag_chat')
def uploadform():
    body_type = "body class='subpage'"
    if len(os.listdir("./static/rag/")) == 0:
        return render_template('rag_chat_upload.html', body_type=body_type, page = "rag")
    else:
        file = os.listdir("./static/rag/")
        pdf_file = file[0]
        return render_template("rag_chat.html", body_type=body_type, pdf_file=pdf_file, page = "rag")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:
        file = request.files['file']

        file.save(f"./static/rag/{file.filename}")

        loader = PDFMinerLoader(f"./static/rag/{file.filename}")
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20
        )
        document_chunks = splitter.split_documents(document)
        # client = chromadb.PersistentClient(settings=Settings(allow_reset=True))
        vector_store = FAISS.from_documents(document_chunks,
                                            embedding=OpenAIEmbeddings())
        vector_store.save_local(folder_path="vector_files")
        session.clear()
        return render_template('/_rag_chat_form.html', pdf_file=file.filename)

    return 'No file uploaded'


@app.route("/get_rag_response", methods=["POST"])
def rag_response():
    user_input = request.form["user_input"]

    if user_input is not None and user_input != "":
        chat_response = rag_agent.chat_rag(user_input)

        return render_template("_rag_chat.html", chat_history=chat_response)


if __name__ == "__main__":
    app.run(debug=True)
