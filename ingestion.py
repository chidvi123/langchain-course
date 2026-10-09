import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore 
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print("INDEX_NAME:", os.getenv("INDEX_NAME"))

    loader=TextLoader(r"C:\Users\PalCh004\LangChain_Udemy\langchain-course\mediumblog1.txt",encoding='utf-8')
    document = loader.load()

    print("Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0) #chunk_size is heiristic ,to keep size small enough to fit context window
    texts=text_splitter.split_documents(document)

    print(f"Created {len(texts)} Chunks")


    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("ingesting...")


    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ['INDEX_NAME'])

    print("Finish..")
