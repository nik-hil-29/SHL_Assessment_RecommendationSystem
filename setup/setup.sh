#!/bin/bash
# Install pip packages
pip install python-dotenv
pip install playwright beautifulsoup4 requests selenium
pip install agentql
pip install nest_asyncio
pip install async
# Install Playwright dependencies and browsers
playwright install-deps
playwright install

# Langchain Dependencies
pip install langchain-google-genai
pip install langchain
pip install langchain-core
pip install langchain-community

# ChromaDb
pip install ChromaDB