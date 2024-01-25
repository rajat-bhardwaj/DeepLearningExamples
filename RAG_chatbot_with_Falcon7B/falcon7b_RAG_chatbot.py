import os
import re

import warnings

warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset

from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

torch.set_default_device("cuda")


class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        text = re.sub(user_pattern, "", text)
        human_pattern = r"\nHuman:"
        text = re.sub(human_pattern, "", text)
        ai_pattern = r"\nAI:"
        return re.sub(ai_pattern, "", text).strip()

    @property
    def _type(self) -> str:
        return "output_parser"


def parse_input(example):
    clean_txt = example["text"].strip().lower().replace("\n", " ")
    story_block = clean_txt.split("introduction")
    title = story_block[0].replace("title:", "").strip()
    return {"title": title, "story": clean_txt}


class BasicRagWithFalcon7B:
    def __init__(self, train: bool):
        self.model_id = "tiiuae/falcon-7b-instruct"
        self.task = "text-generation"
        self.model_settings = {
            "bos_token_id": 1,
            "eos_token_id": 11,
            "max_new_tokens": 500,
            "pad_token_id": 11,
            "repetition_penalty": 1.7,
            "temperature": 0.3,
        }

        self.train = train
        self.llm = self.get_llm()
        self.vector_store = self.get_vector_store()

    def get_llm(self):
        llm = HuggingFaceHub(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            repo_id=self.model_id,
            task=self.task,
            model_kwargs=self.model_settings,
        )
        return llm

    def get_memory(self):
        memory = ConversationBufferWindowMemory(
            memory_key="history", k=6, return_only_outputs=True
        )

        return memory

    def example_chain(self):
        template = """ 
        You are a polite and helpful chatbot that generate stories and answer questions.
        And if you don't know the answer to a question, you can truthfully says you does not know.
        
        Current conversation:
        {history}
        Human: {input}
        AI:""".strip()

        prompt = PromptTemplate(input_variables=["history", "input"], template=template)

        chain = ConversationChain(
            llm=self.llm,
            memory=self.get_memory(),
            prompt=prompt,
            verbose=False,
            output_parser=CleanupOutputParser(),
        )

        return chain

    def load_example_ds(self):
        tiny_story_ds = load_dataset(
            "nampdn-ai/tiny-strange-textbooks",
            data_files="data_part_0.parquet",  # limiting to one file for this example
            split="train",
            cache_dir=".",
            download_mode="force_redownload",
        )

        tiny_story_ds = tiny_story_ds.map(
            parse_input
        )  # add batch to make the computation faster

        return tiny_story_ds

    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": False},
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        return embeddings

    def get_vector_store(self):
        if self.train:
            ds = self.load_example_ds()

            vector_store = FAISS.from_texts(
                texts=ds["story"],
                embedding=self.get_embeddings(),
            )
            vector_store.save_local("FAISS_index_tiny_stories")
        else:
            vector_store = FAISS.load_local(
                "FAISS_index_tiny_stories", self.get_embeddings()
            )

        return vector_store

    def get_retrieval_chain(self):
        qa_template = """Use the given context to answer the question or write a story. 
        Do not answer anything outside of this context.
        If you don't know the answer, just say that you don't know. Never hallucinate or repeat the answers.
        Keep the answer as concise as possible.
        Context: {context}
        Question: {question}
        Answer:""".strip()

        qa_prompt_template = PromptTemplate.from_template(qa_template)

        # Define the RetrievalQ&A chain
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vector_store.as_retriever(  # search_type="similarity_score_threshold",
                search_kwargs={"k": 3}  # "score_threshold": 0.5,
            ),
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt_template},
        )

        return qa_chain
