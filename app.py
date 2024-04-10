import warnings
warnings.filterwarnings("ignore")

import os
import textwrap

import langchain
from langchain.llms import HuggingFacePipeline

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

print(langchain.__version__)
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
class CFG:
    model_name = 'falcon' 
def get_model(model = CFG.model_name):

    print('\nDownloading model: ', model, '\n\n')

    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2")
        
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
    max_len = 1024
    task = "text-generation"
    T = 0


    return tokenizer, model, max_len, task, T
  %%time

tokenizer, model, max_len, task, T = get_model(CFG.model_name)
pipe = pipeline(
    task=task,
    model=model,
    tokenizer=tokenizer,
    max_length=max_len,
    temperature=T,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)
from transformers import pipeline
# Save the pipeline model
pipe.save_pretrained("Saved_pipeline")
!zip -r model.zip /kaggle/working/Saved_pipeline
!zip -r model.zip /kaggle/working/Saved_pipeline
%%time

loader = DirectoryLoader('/kaggle/input/dataset/', 
                         glob="./*.pdf",
                         loader_cls=PyPDFLoader,
                         show_progress=True,
                         use_multithreading=True)

documents = loader.load()
!zip -r file.zip /kaggle/input/output/cse-syllabus-vectordb-chroma
! unzip file.zip
%%time

load_directory = '/kaggle/working/kaggle/input/output/cse-syllabus-vectordb-chroma'

 ### download embeddings model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                       model_kwargs={"device": "cuda"})

vectordb = Chroma(persist_directory=load_directory, embedding_function=instructor_embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3,}) # "search_type" : "similarity"

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=True,
                                       verbose=False)
qa_chain
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
#     print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
def llm_ans(query):
    llm_response = qa_chain(query)
    ans = process_llm_response(llm_response)
    return ans
