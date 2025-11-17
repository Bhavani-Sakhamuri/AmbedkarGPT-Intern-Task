import os
from langchain.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Step 1: Load and split the document
def load_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(
    separator="\n",       # or "\n\n" if your text has paragraph breaks
    chunk_size=500,
    chunk_overlap=100)

    #splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Step 2: Create vector store using ChromaDB and HuggingFace embeddings
def create_vector_store(docs, persist_directory="chroma_db"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

# Step 3: Set up the RAG pipeline using LangChain's RetrievalQA
def setup_rag_pipeline(vectordb):
    retriever = vectordb.as_retriever()
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Step 4: Run the pipeline
def main():
    file_path = "speech.txt"
    print("🔍 Loading and processing documents...")
    docs = load_documents(file_path)

    print("📦 Creating vector store...")
    vectordb = create_vector_store(docs)

    print("🤖 Setting up RAG pipeline...")
    qa_chain = setup_rag_pipeline(vectordb)

    print("✅ Ready! Ask questions about the document.")
    while True:
        query = input("\n🧠 Your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = qa_chain.invoke(query)
        print(f"\n📣 Answer: {response}")

if __name__ == "__main__":

    main()
