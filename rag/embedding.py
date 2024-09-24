import json
import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class EmbeddingManager:
    def __init__(
        self, data_file: str, db_directory: str, model_name: str = 'nomic-embed-text'
    ):
        self.file_path = data_file
        self.db_dir = db_directory
        self.embeddings = OllamaEmbeddings(model=model_name)

    def load_faqs(self) -> list[Document]:
        """Load FAQs from JSON file and convert to Document objects."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f'The file {self.file_path} does not exist. Please check the path.'
            )

        with open(self.file_path, 'r') as file:
            data = json.load(file)

        docs = []
        for chunk in data.get('questions', []):
            if 'question' in chunk and 'answer' in chunk:
                text = f"Question: {chunk['question']}\nAnswer: {chunk['answer']}"
                docs.append(Document(page_content=text, metadata={'source': 'FAQs'}))
            else:
                logging.warning(f'Invalid chunk: {chunk}')

        logging.info(f'Loaded {len(docs)} document chunks.')
        return docs

    def create_vector_store(self, docs: list[Document], store_name: str):
        """Create and persist vector store."""
        persistent_directory = os.path.join(self.db_dir, store_name)

        if not os.path.exists(persistent_directory):
            logging.info(f'Creating vector store {store_name}')
            Chroma.from_documents(
                docs, self.embeddings, persist_directory=persistent_directory
            )
            logging.info(f'Finished creating vector store {store_name}')
        else:
            logging.info(
                f'Vector store {store_name} already exists. No need to initialize.'
            )


def main():
    # Define paths and model name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'faqs.json')
    db_directory = os.path.join(current_dir, 'db')

    # Initialize EmbeddingManager
    manager = EmbeddingManager(data_file=data_file, db_directory=db_directory)

    # Load FAQs and create vector store
    docs = manager.load_faqs()
    manager.create_vector_store(docs, 'chroma_db_ollama')

    logging.info('Embedding process completed.')


if __name__ == '__main__':
    main()
