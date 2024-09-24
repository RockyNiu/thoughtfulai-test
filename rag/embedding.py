import json
import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'faqs.json')
db_dir = os.path.join(current_dir, 'db')


def load_faqs(file_path) -> list[Document]:
    """Load FAQs from JSON file and convert to Document objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'The file {file_path} does not exist. Please check the path.'
        )

    with open(file_path, 'r') as file:
        data = json.load(file)

    docs = []
    for chunk in data.get('questions', []):
        if 'question' in chunk and 'answer' in chunk:
            text = f"Question: {chunk['question']}\nAnswer: {chunk['answer']}"
            docs.append(Document(page_content=text, metadata={'source': 'FAQs'}))
        else:
            logging.warning(f'Invalid chunk: {chunk}')

    logging.debug(f'Number of document chunks: {len(docs)}')
    return docs


def create_vector_store(docs, embeddings, store_name):
    """Create and persist vector store."""
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        logging.debug(f'Creating vector store {store_name}')
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        logging.debug(f'Finished creating vector store {store_name}')
    else:
        logging.debug(
            f'Vector store {store_name} already exists. No need to initialize.'
        )


def main():
    # Load FAQs
    docs = load_faqs(file_path)

    embeddings = OllamaEmbeddings(
        model='nomic-embed-text',
    )

    # Create vector store
    create_vector_store(docs, embeddings, 'chroma_db_ollama')
    logging.debug('Embedding completed.')


if __name__ == '__main__':
    main()
