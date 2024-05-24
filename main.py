import os
import logging
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from core import retrieval_qa_pipline
import utils
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.callbacks.manager import CallbackManager
from configs import (
    MODELS_PATH,
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME
)
from multi_project_support import (
    get_PERSIST_DIRECTORY,
    get_SOURCE_DIRECTORY,
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def load_documents_and_split(args):
    # Multi projects support
    SOURCE_DIRECTORY = get_SOURCE_DIRECTORY(args.proj_name)
    PERSIST_DIRECTORY = get_PERSIST_DIRECTORY(args.proj_name)

    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = utils.load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = utils.split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": args.device_type},
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

def main(args):
    if args.task == "document_processing":
        logging.info(f"Running document processing on: {args.device_type}")
        load_documents_and_split(args)
    elif args.task == "start_qa_retrieval":
        logging.info(f"Starting QA retrieval on: {args.device_type}")
        logging.info(f"Display Source Documents set to: {args.show_sources}")
        logging.info(f"Use history set to: {args.use_history}")

        if not os.path.exists(MODELS_PATH):
            os.mkdir(MODELS_PATH)

        qa = retrieval_qa_pipline(args.device_type, args.use_history, args.proj_name, promptTemplate_type=args.model_type)

        while True:
            query = input("\nEnter a query: ")
            if query == "exit":
                break
            
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            if args.show_sources:
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in docs:
                    print("\n> " + document.metadata["source"] + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
            
            if args.save_qa:
                utils.log_to_csv(query, answer)
    else:
        logging.error("Invalid task provided. Please select either 'document_processing' or 'start_qa_retrieval'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine document processing and QA retrieval.")
    parser.add_argument("task", choices=["document_processing", "start_qa_retrieval"], help="Task to perform.")
    parser.add_argument("--device_type", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep",
                                "hip", "ve", "fpga", "ort", "xla", "lazy", "vulkan", "mps",
                                "meta", "hpu", "mtia"],
                    help="Device to run on. (Default is cuda)")
    parser.add_argument("--show_sources", "-s", action="store_true",
                    help="Show sources along with answers (Default is False)")
    parser.add_argument("--use_history", "-u", action="store_true",
                    help="Use history (Default is False)")
    parser.add_argument("--model_type", default="llama",
                    choices=["llama", "mistral", "non_llama"],
                    help="model type, llama, mistral or non_llama")
    parser.add_argument("--save_qa", action="store_true",
                    help="whether to save Q&A pairs to a CSV file (Default is False)")
    parser.add_argument("--proj_name", default="NewProj",
                    help="Project name. Default is (NewProj)")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )

    main(args)
