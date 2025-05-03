import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
import pinecone
import os

# --- ユーザー設定 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX")

st.title("📚 BookQA AI チャットボット")

question = st.text_input("質問を入力してください")

if question:
    # Pinecone 初期化
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    # インデックスを取得
    index = pinecone.Index(INDEX_NAME)

    # Embeddingsを作成
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # ベクトルストアを作成（langchain_pineconeを使用）
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    # AI モデルと QA チェーンを作成
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    # 質問を実行
    answer = qa.run(question)

    # 結果を表示
    st.markdown("### 🧠 AIからの回答")
    st.write(answer)

    st.markdown("---")
    st.caption("（参考文献：Pineconeから検索された書籍データ）")
