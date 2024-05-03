import os
os.environ["OPENAI_API_KEY"] = "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"
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def download_history(query_history, result_history):
    history_content = ""
    for query, result in zip(query_history, result_history):
        history_content += f"Query: {query}\nResult: {result}\n\n"

    st.download_button(
        label="Download History",
        data=history_content,
        file_name="query_history.txt",
        mime="text/plain",
    )

  
# image_path = os.path.join('C:', 'Users', 'DELL I7', 'Documents', 'Python Project', 'Streamlit Dev', 'Final project', 'materials', 'PNG HIC.png')
# st.image(image_path)

def main():
    st.title("OS AI TUTORING ASSISTANT")

    # Initialize session state to store query history and results
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
        st.session_state.result_history = []

    # Sidebar with add icon
    with st.sidebar:
        st.markdown("""
            <style>
                .sidebar-add-icon {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    font-size: 24px;
                    color: green;
                    cursor: pointer;
                }
            </style>
        """, unsafe_allow_html=True)
        st.markdown("""
            <span class="sidebar-add-icon" onclick="alert('Add icon clicked!')">&#10010;</span>
        """, unsafe_allow_html=True)

        st.sidebar.title("History")
        for i, query in enumerate(st.session_state.query_history):
            if st.sidebar.button(query, key=f"history_{i}"):
                st.write(st.session_state.result_history[i])

    # Upload PDF file
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file is not None:
        # Step 04: Extracting Text from the PDF Document using PDF Reader
        reader = PdfReader(pdf_file)

        # Step 05: Read Data From the PDF File and put it into a variable raw_text
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        # Step 06: Split Text into Smaller Chunks
        textsplitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = textsplitter.split_text(raw_text)

        # Step 07: Download Embeddings from OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY")), chain_type='stuff')

        # Get user query
        query = st.text_input("Enter your query")
        if query:
            # Add query to history
            st.session_state.query_history.append(query)

            # Perform question answering
            docs = docsearch.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            st.session_state.result_history.append(result)
            st.write(result)

    # Download history button
    download_history(st.session_state.query_history, st.session_state.result_history)

if __name__ == "__main__":
    main()