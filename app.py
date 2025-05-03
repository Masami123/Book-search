import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# --- ユーザー設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX")

st.title("📚 BookQA AI チャットボット")

question = st.text_input("質問を入力してください")

if question:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    answer = qa.run(question)

    st.markdown("### 🧠 AIからの回答")
    st.write(answer)

    st.markdown("---")
    st.caption("（参考文献：Pineconeから検索された書籍データ）")
