import os
import logging
import re
import requests     
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.llms.openai import OpenAI as OpenAILamaindex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb
import Stemmer


def scrape_jina_ai(url: str) -> str:
    """
    Scrapes web page content using Jina AI.

    Args:
        url (str): The web page to scrape

    Returns:
        str: text content scraped from the URL.
    """
    response = requests.get("https://r.jina.ai/" + url)
    return response.text

def parse_markdown_text(markdown_text: str) -> str:
    """
    Cleans a markdown text by removing block markers.

    Args:
        markdown_text (str): The markdown text to be cleaned.

    Returns:
        str: The cleaned text with block markers removed.
    """
    # Remove block markers
    cleaned_text = re.sub(r'```.*?\n|```', '', markdown_text)
    cleaned_text = cleaned_text.strip() 
    return cleaned_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/prepare_vector_keyword_indexes_2.log"),
    ]
)

load_dotenv()  # Load environment variables from .env file
logging.info("Environment variables loaded.")

# Initialize and configure the OpenAI embedding and language models, setting them in the global settings for application-wide use.
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAILamaindex("gpt-4o-mini", temperature=0.7)
Settings.llm = llm
Settings.embed_model = embed_model

markdown_parser = MarkdownNodeParser()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# list to consolidate all nodes for BM25Retriever use
all_nodes = []


# ---------------------- scrape household support data ------------------------ #
# Scrape data from website
response = scrape_jina_ai("https://www.mof.gov.sg/singaporebudget/resources/support-for-households")
# Save scraped data to file
with open('/home/ed/htx0/data/scrapes/household_support.txt', 'w', encoding='utf-8') as f:
    f.write(response)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Convert scraped text to markdown structure
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a top expert in converting text into markdown formats."},
    {"role": "user", "content": "Convert the following text into markdown format: " + response}
  ],
  temperature=0.0
)
markdown_response = completion.choices[0].message.content

# Clean markdown text
parsed_markdown_response = parse_markdown_text(markdown_response)

# Save cleaned scraped data to file
with open('/home/ed/htx0/data/scrapes/household_support_markdown.txt', 'w', encoding='utf-8') as f:
    f.write( parsed_markdown_response )


# ----------------------- Parse household support data ------------------------ #
# Chunk into nodes
updated_documents_household = Document( text=parsed_markdown_response )
nodes_household_markdown = markdown_parser.get_nodes_from_documents([updated_documents_household])
logging.info(f"Number of chunk nodes for scraped household support data: {len(nodes_household_markdown)}")

# Consolidate nodes
all_nodes.extend(nodes_household_markdown)
logging.info(f"Total nodes consolidated after parse scraped household support data: {len(all_nodes)}")


# ---------------------- Vector store setup and populate ---------------------- #
chroma_collection = chroma_client.get_or_create_collection("budget_household_collection")

# Prepare Chroma vector store
vector_store_household = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context_household = StorageContext.from_defaults(vector_store=vector_store_household)

if not os.path.exists("budget2024_household_chroma"): 
    # Create index 'V2'
    index_household = VectorStoreIndex(
        nodes=nodes_household_markdown,
        storage_context=storage_context_household,
        embed_model=embed_model
    )
    index_household.storage_context.persist(persist_dir="./budget2024_household_chroma")
    logging.info(f"Added nodes into vector store for: scraped household_support.txt")
else:
    ctx = StorageContext.from_defaults(persist_dir="./budget2024_household_chroma")
    index_household = load_index_from_storage(ctx)


#---------------------- Keyword index setup and populate --------------------- #
## BM25Retriever does not support dynamically adding of new nodes 

# Create text store index 'T2'
retriever_bm25_budget_household = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)
if retriever_bm25_budget_household:
    logging.info(f"Added all nodes into keyword store")
    retriever_bm25_budget_household.persist("./budget2024_household_keyword")