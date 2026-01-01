"""Configuration for LLM and embeddings."""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sk import my_sk


# LLM Configuration
llm = ChatOpenAI(model="gpt-4o-mini", api_key=my_sk, temperature=0)

# Embeddings Configuration
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=my_sk)

