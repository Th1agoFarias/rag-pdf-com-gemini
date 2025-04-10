import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ✅ Corrigido aqui
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

load_dotenv()

def main():
    st.set_page_config(page_title="📊 Pergunte ao seu PDF")
    st.header("🔍 RAG com PDF e Gemini")

    pdf = st.file_uploader("Faça upload do seu PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""
            text += extracted_text + "\n"

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Pergunte sobre seu PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            if docs:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(input_documents=docs, question=user_question)

                st.write(response)
            else:
                st.warning("Nenhuma informação relevante encontrada no PDF.")

if __name__ == '__main__':
    main()
