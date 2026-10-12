import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # get to know about this 
from operator import itemgetter 
#uv add langchain-pinecone langchain-huggingface

load_dotenv()

print("Initializing components...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],embedding=embeddings
)


retriever= vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context 
    {context}

    Question:{question}

    Provide a detailed answer :
    """
)


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


###############################################################
# IMPLEMENTATION 1:WITHOUT LCEL (SIMPLE FUNCTION BASED APPROACH)
###############################################################

def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents , formats them , and generates a response 

    Limitations:
    - Manaual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains 
    - More verbose and error-prone 
    """

    #step 1: Retrive relevant docs

    docs = retriever.invoke(query)

    #step 2 
    context = format_docs(docs)

    #step 3 
    messages=prompt_template.format_messages(context=context,question=query)

    #step 4
    response = llm.invoke(messages)

    #step 5 
    return response.content

###################################################################
# IMPLEMENATION 2: WITH LCEL - BETTER APPROACH
###################################################################

def create_retrieval_chain_with_lcel():
    """
    creates a retrieval chain using LCEL.
    Returns a chain that can be invoked with {"question":"..."}.
    """

    retrieval_chain =(
        #format_docs # normal fucntion , not a runnable one , but langchain chnages it to (format_docs -> runnablelambda(format_docs))
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        |
        prompt_template
        |
        llm
        |
        StrOutputParser()
    )

    return retrieval_chain

if __name__=="__main__":
    print("Retrieving...")

    #query 
    query="what is Pinecone in Machine Learning"


    ################################################################
    # Option 0 : RAW INVOCATION WITHOUT RAG
    print("\n"+"="*70)
    print("IMPLEMENTATION O: RAW LLM INVOCATION (NO RAG)")
    print("="*70)
    result_raw=llm.invoke([HumanMessage(content=query)])
    print("\n Answer :")
    print(result_raw.content)
    ##############################################################

    #############################################################
    # OPTION 1 : USE IMPLEMENTATION WITHOUT LCEL
    print("\n"+"="*70)
    print("IMPLEMENTATION 1 : WITHOUT LCEL")
    print("="*70)
    result_without_lcel=retrieval_chain_without_lcel(query)
    print("\n Answer")
    print(result_without_lcel)
    ##############################################################



    #############################################################
    # OPTION 2: USE IMPLEMENTATION 2 : WITH LCEL - BETTER APPROACH

    print('\n'+'='*70)
    print("IMPLEMENTATION 2 : WITH LCEL - BETTER APPROACH")
    print("="*70)

    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question":query})

    print("\n Answer :")
    print(result_with_lcel)
    #############################################################


