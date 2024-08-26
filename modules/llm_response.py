from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
from transformers import   pipeline,AutoModelForQuestionAnswering, AutoTokenizer , AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq

import torch
import torch.nn as nn
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMResponseError(Exception):
    pass
def verify_answer_in_context(context, answer):
    if answer.lower() not in context.lower():
        return "The information is not available in the provided context."
    return answer

def get_response_from_llm(doc_search, user_question):
    try:
        # repo_id = "google/flan-t5-large"
        # llm = HuggingFaceEndpoint(huggingfacehub_api_token = os.getenv("HUGGING_FACE_HUB_API_KEY"),
        #     repo_id=repo_id, temperature=5, 
        #                           max_length=64)
        # docs = doc_search .similarity_search(user_question)
        # content = "\n".join([x.page_content for x in docs])

        # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=r"C:\Users\Umar\Videos\AnyDesk", device_map="auto")
        # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", cache_dir=r"C:\Users\Umar\Videos\AnyDesk")
        # pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        # llm = HuggingFacePipeline(
        # pipeline = pipe,
        # model_kwargs={"temperature": 0.01, "max_length": 512},
        # )
      
        # #   Load the QA chain using load_qa_chain
        # qa_chain = load_qa_chain(
        #     llm=llm,
        #     chain_type="stuff",
        #     # prompt=QA_CHAIN_PROMPT
        # )

        # # Retrieve documents and prepare the input for the chain
        # docs = doc_search.similarity_search(user_question)
        # context = "\n".join([doc.page_content for doc in docs])
        # print("context",context)
        # inputs = {
        #     "input_documents": docs,
        #     "question": user_question
        # }
        
        # # Get the result from the QA chain
        # result = qa_chain.run(inputs)
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        chat = ChatGroq(temperature=2, groq_api_key='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',model_name="mixtral-8x7b-32768")     #Gemma-7b-it    mixtral-8x7b-32768
        qa_chain = RetrievalQA.from_chain_type(   
        llm=chat,   
        chain_type="stuff",   
        retriever=doc_search.as_retriever(),
        return_source_documents= True,  
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
        )
        result = qa_chain ({ "query" : user_question}) 
                
        # print(result["result"])
        # verified_result = verify_answer_in_context(context, result)
        
        # if verified_result == "":
            # return "No context available"
        # print(verified_result)
        return result["result"]
    except KeyError as ke:
        raise LLMResponseError(f"KeyError occurred while accessing environment variable: {str(ke)}")
    except Exception as e:
        raise LLMResponseError(f"Error getting response from LLM: {str(e)}")