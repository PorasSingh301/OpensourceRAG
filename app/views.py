from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

model = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="llama2")

parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

loader = PyPDFLoader(r"C:\Users\Poras\Envs\rag_env\rag\Resume-Poras-Singh.pdf")
pages = loader.load_and_split()

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)

retriever = vectorstore.as_retriever()

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)


@api_view(http_method_names=["POST"])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def ask_resume(request):
    response = chain.invoke({'question': request.data["question"]})
    return Response({"answer": response})
