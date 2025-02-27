from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain.schema import Generation
import torch

# Load your tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("NLPGenius/GovGPT-llama3")
model = AutoModelForCausalLM.from_pretrained("NLPGenius/GovGPT-llama3", 
                                             #torch_dtype=torch.bfloat16, 
                                             device_map="auto",)


import transformers
from transformers import pipeline
llm_chain = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.3,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    do_sample=True,
    max_length=2000,  # max number of tokens to generate in the output
    truncation=True,
    repetition_penalty=1.1  # without this output begins repeating
)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=llm_chain)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Define the chunking and embedding strategy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedding_model = HuggingFaceEmbeddings()  # Adjust if you have a specific embedding model

import os
import rarfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Function to create FAISS index
def create_faiss_index(docs):
    texts = text_splitter.split_documents(docs)  # Ensure `text_splitter` is defined in your script
    vector_store = FAISS.from_documents(texts, embedding_model)
    return vector_store

# Function to fetch and process PDF files
def fetch_and_process_pdfs(pdf_paths):
    all_documents = []
    for pdf_file in pdf_paths:
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

# Function to create RAG pipeline with provided PDF paths
def create_rag_pipeline(pdf_paths):
    # Fetch and process PDF files
    documents = fetch_and_process_pdfs(pdf_paths)

    # Create vector store
    vector_store = create_faiss_index(documents)
    retriever = vector_store.as_retriever()

    return retriever

# Specify the PDF paths
pdfs_paths = [
    "Esta Code 2024 final2.pdf", 
    "Executive Handbook 2024 PDF.pdf", 
    "Rules of Business, 1985 (Updated 2024).pdf", 
]


# Usage example
retriever = create_rag_pipeline(pdfs_paths)


from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import re

# # Define a custom output parser to extract only the answer
# class AnswerOutputParser(BaseOutputParser):
#     def parse(self, text: str) -> str:
#         # Extract everything after "Answer:"
#         if "Answer:" in text:
#             return text.split("Answer:")[-1].strip()
#         return "I don't know."

# Function to perform model inference
import re
from fuzzywuzzy import fuzz  # For relevance matching

def is_context_relevant(context, question, threshold=40):
    """
    Checks if the retrieved context is relevant to the question using fuzzy matching.
    Returns True if relevant, False otherwise.
    """
    context_snippet = " ".join(context.split()[:100])  # Use only first 100 words for efficiency
    relevance_score = fuzz.partial_ratio(context_snippet.lower(), question.lower())

    return relevance_score >= threshold  # Only accept if score is above threshold

def model_inference(retriever, question, llm):
    # Validate question: Reject if too short or empty
    if not question.strip() or len(question.strip()) < 5:
        return "Answer:\n        The question is invalid or lacks sufficient detail."

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # If no relevant documents are found, reject the query
    if not retrieved_docs:
        return "Answer:\n        The answer is not found in the context provided.\nReferences: No references found."

    # Extract context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs]).strip()

    # **New: Check if the retrieved context is relevant to the question**
    if not is_context_relevant(context, question):
        return "Answer:\n        The answer is not found in the context provided.\nReferences: No references found."

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""\
        You are a highly accurate and reliable assistant. Follow these strict rules:

        1. **Use Only the Provided Context** – Base your answer strictly on the given context. Do not infer or generate extra details.
        2. **Reject Invalid or Unrelated Questions** – If the question does not relate to the context, respond with: "The answer is not found in the context provided."
        3. **Ensure Context Relevance** – Answer only if the exact information exists in the context.
        4. **No Hallucination** – Do not assume, summarize, or infer beyond what is explicitly stated.
        5. **Concise and Relevant Answers** – Avoid redundancy. Provide only the most accurate response.

        Context:
        {context}

        Query:
        {question}

        Answer:
        """
    )

    # Define processing chain
    rag_chain = (
        RunnablePassthrough.assign(context=lambda _: context, question=lambda _: question)
        | prompt
        | llm
    )

    # Invoke the chain and get the response
    response = rag_chain.invoke({}).strip()

    # Extract only the final "Answer:" section
    match = re.findall(r"Answer:\s*(.*)", response, re.DOTALL)
    final_answer = "Answer:\n        " + match[-1].strip() if match else "Answer:\n        The answer is not found in the context provided."

    # Extract only the first relevant reference
    references = (
        f"Source: {retrieved_docs[0].metadata['source']}, Page: {retrieved_docs[0].metadata.get('page_label', 'Unknown')}"
        if retrieved_docs else "No references found."
    )

    return f"{final_answer}\nReferences: {references}"

# # Example usage:
# formatted_output = model_inference(retriever, "What is the main purpose of the Khyber Pakhtunkhwa Arms Act, 2013?", llm)

#response = model_inference(retriever, "Why was a new agriculture policy needed in Khyber Pakhtunkhwa?")



from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

# Mock chatbot function (replace with your actual pipeline)
def chatbot_response(query):
    try:
        # Call your actual RAG pipeline here
        response = model_inference(retriever, query, llm)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Route for chatbot responses
@app.route('/', methods=['GET','POST'])
def chat():
    
    # Try to get JSON data
    if request.content_type == 'application/json':
        data = request.get_json(silent=True)
        user_query = data.get("query", "").strip() if data else ""
    else:
        # Fallback to form-data
        user_query = request.form.get("query", "").strip()
    
    if not user_query:
        return jsonify({"error": "Please provide a valid query."}), 400

    # Get chatbot response
    bot_reply = chatbot_response(user_query)
    return jsonify({"response": bot_reply})

# Run the Flask app
if __name__ == '__main__':
    app.run()
