# ThoughtfulAI Test
By using ollama to host LLAMA-3.1 and nomic-embed-text, this project aims to create a simple CLI application that can answer questions based on the given faqs.

## Setup
### Requirements
- Install [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
- Install [poetry](https://python-poetry.org/docs/)

### Installation
- install python packages
   ```bash
   pyenv install 3.12
   pyenv global 3.12
   poetry env use python3.12
   poetry install
   poetry shell
   ```
- install [ollama](https://ollama.com/)
   ```
   ollama pull llama3.1 # LLM
   ollama pull nomic-embed-text # embedding
   ```

### Run
- load data into chroma db
   ```bash
   python rag/embedding.py
   ```
- run the server
   ```bash
   python rag/agent.py
   ```
- for VSCode users, you can use the provided launch configuration to run: select the file, and click on the play button