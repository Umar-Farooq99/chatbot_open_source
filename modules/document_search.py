from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

class DocumentSearchError(Exception):
    pass

def create_document_search(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size = 800,
            chunk_overlap  = 0,
            length_function = len,
            add_start_index= True,
        )
        texts = text_splitter.split_text(text) 
        print("/////////////",texts)              # here used split_text before split_documents
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_texts(texts=texts, embedding=embeddings)  
        # db = Chroma.from_texts(texts,embeddings)

        # return db                     # here used from_texts before this
    except Exception as e:
        raise DocumentSearchError(f"Error creating document search: {str(e)}")