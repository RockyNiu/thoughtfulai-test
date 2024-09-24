import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings


class ChatAgent:
    def __init__(self):
        self.persistent_directory = self._get_persistent_directory()
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')
        self.db = self._load_vector_store()
        self.retriever = self._create_retriever()
        self.llm = ChatOllama(model='llama3.1')
        self.rag_chain = self._create_rag_chain()

    def _get_persistent_directory(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'db', 'chroma_db_ollama')

    def _load_vector_store(self) -> Chroma:
        return Chroma(
            persist_directory=self.persistent_directory,
            embedding_function=self.embeddings,
        )

    def _create_retriever(self):
        return self.db.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    def _create_rag_chain(self):
        contextualize_q_prompt = self._create_contextualize_q_prompt()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        qa_prompt = self._create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def _create_contextualize_q_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            'Given a chat history and the latest user question '
            'which might reference context in the chat history, '
            'formulate a standalone question which can be understood '
            'without the chat history. Do NOT answer the question, just '
            'reformulate it if needed and otherwise return it as is.'
        )
        return ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )

    def _create_qa_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            'You are an assistant for question-answering tasks. Use '
            'the following pieces of retrieved context to answer the '
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            'concise.'
            '\n\n'
            '{context}'
        )
        return ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )

    def chat(self):
        print("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history: list[HumanMessage | AIMessage] = []
        while True:
            query = input('You: ')
            if query.lower() == 'exit':
                break
            result = self.rag_chain.invoke(
                {'input': query, 'chat_history': chat_history}
            )
            print(f"AI: {result['answer']}")
            chat_history.extend(
                [HumanMessage(content=query), AIMessage(content=result['answer'])]
            )


if __name__ == '__main__':
    agent = ChatAgent()
    agent.chat()
