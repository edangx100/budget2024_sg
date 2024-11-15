import os
import logging
import re
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI as OpenAILamaindex
from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb
import Stemmer


def get_data_files(data_dir) -> list[str]:
    """Retrieve a list of file paths from a specified directory.

    Args:
        data_dir (str): The directory path from which to retrieve file paths.

    Returns:
        list[str]: A list of file paths found in the specified directory.
    """
    files = []
    logging.info(f"Retrieving data files from directory: {data_dir}")
    for f in os.listdir(data_dir):
        fname = os.path.join(data_dir, f)
        if os.path.isfile(fname):
            files.append(fname)
            logging.debug(f"File found: {fname}")  # Log each file found
    logging.info(f"Total files retrieved: {len(files)}")
    return files

def concatenate_pages(documents: list[Document]) -> str:
    """Concatenates the text content of all pages in a list of documents.
    Args:
    documents_with_instruction: A list of document objects, where each object
        has a 'text' attribute containing the text content of a page.

    Returns:
    A string containing the concatenated text content of all pages.
    """
    concatenated_text = ""
    for page in documents:
        concatenated_text += page.text
    return concatenated_text


# ----------------------------------------------------------------------------- #

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/prepare_vector_keyword_indexes_1.log"),
    ]
)

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded.")

MAIN_DATA_DIR = os.path.join(os.path.expanduser("~"), "htx0", "data")
ANNEX_DATA_DIR = os.path.join(os.path.expanduser("~"), "htx0", "data", "annex")
main_filepaths = get_data_files(MAIN_DATA_DIR)
annex_filepaths = get_data_files(ANNEX_DATA_DIR)

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


# --------------------- Parse fiscal position annex file ---------------------- #
# Parse document
documents_fiscal = LlamaParse(
    result_type="markdown",
    parsing_instruction="""
    * You are given documents on Budget Statement For Budget 2024.
    * Ensure that tables are formatted in Markdown. If any cells are merged within the tables, this should also be reflected in the Markdown format.
    * Respond in markdown format.
    """,
    parse_all_pages=True,
    ).load_data(annex_filepaths[10])
logging.info(f"Number of pages for annexh2_fiscal_position.pdf: {len(documents_fiscal)}")

all_pages_fiscal = concatenate_pages(documents_fiscal)

# Remove markdown wrappers
updated_all_pages_fiscal = re.sub(r"(```markdown|``````markdown|```|``````)", "\n\n", all_pages_fiscal)

# Chunk into nodes
updated_documents_fiscal = Document(text=updated_all_pages_fiscal)
nodes_fiscal_markdown = markdown_parser.get_nodes_from_documents([updated_documents_fiscal])
logging.info(f"Number of chunk nodes for annexh2_fiscal_position.pdf: {len(nodes_fiscal_markdown)}")

# Consolidate nodes
all_nodes.extend(nodes_fiscal_markdown)
logging.info(f"Total nodes consolidated after parse annexh2_fiscal_position.pdf: {len(all_nodes)}")


# ----------------- Initial vector store setup and populate ------------------- #
chroma_budget_collection = chroma_client.get_or_create_collection("budget_collection")

# Prepare Chroma vector store
vector_store_budget_collection = ChromaVectorStore(chroma_collection=chroma_budget_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store_budget_collection)

if not os.path.exists("budget2024_chroma"): 
    # Create index 'V1'
    index_budget = VectorStoreIndex(
        nodes=nodes_fiscal_markdown,
        storage_context=storage_context,
        embed_model=embed_model
    )
    index_budget.storage_context.persist(persist_dir="./budget2024_chroma")
    logging.info(f"Added nodes into vector store for: annexh2_fiscal_position.pdf")
else:
    ctx = StorageContext.from_defaults(persist_dir="./budget2024_chroma")
    index_budget = load_index_from_storage(ctx)


# ------------ Parse remaining annex files and update vector store ------------ #
annex_all_filepaths_set = set(annex_filepaths)

# exclude fiscal annex which was processed in earlier step
annex_filepaths_set = {file for file in annex_all_filepaths_set if 'annexh2_fiscal_position.pdf' not in file}

# Process each annex file
for i, annex_filepath in enumerate(annex_filepaths_set):
    annex_filename = os.path.basename(annex_filepath)
    
    # Parse document
    documents_annex = LlamaParse(
        result_type="markdown",
        parsing_instruction="""
        * You are given documents on Budget Statement For Budget 2024.
        * Ensure that tables are formatted in Markdown. If any cells are merged within the tables, this should also be reflected in the Markdown format.
        * Respond in markdown format.
        """,
        parse_all_pages=True,
        ).load_data(annex_filepath)
    logging.info(f"Number of pages for {annex_filename}: {len(documents_annex)}")

    # Concatenate pages
    all_pages_annex = concatenate_pages(documents_annex)

    # Remove markdown wrappers
    updated_all_pages_annex = re.sub(r"(```markdown|``````markdown|```|``````)", "\n\n", all_pages_annex)

    # Chunk into nodes
    document_annex = Document(text=updated_all_pages_annex)
    nodes_annex_markdown = markdown_parser.get_nodes_from_documents([document_annex])
    logging.info(f"Number of chunk nodes for {annex_filename}: {len(nodes_annex_markdown)}")

    # Consolidate nodes
    all_nodes.extend(nodes_annex_markdown)
    logging.info(f"Total nodes consolidated after parse {annex_filename}: {len(all_nodes)}")

    # Add new nodes to the existing index
    index_budget.insert_nodes(nodes_annex_markdown)
    logging.info(f"Added nodes into vector store for: {annex_filename}")

if i == len(annex_filepaths_set) - 1:
    logging.info("All annex files processed successfully.")
else:
    logging.info(f"Not all annex files processed.")


# ------------------ Parse main files and update vector store ----------------- #
# main files include (1) fy2024_budget_debate_round_up_speech.pdf (2) fy2024_budget_statement.pdf

# Process each main file
for j, main_filepath in enumerate(main_filepaths):
    main_filename = os.path.basename(main_filepath)

    # Parse document
    documents_main = LlamaParse(
        result_type="markdown",
        parsing_instruction="""
        * Parse text word for word.
        * Exclude parsing phrases of form: 'Page x of y'
        * Ensure that tables are formatted in Markdown. If any cells are merged within the tables, this should also be reflected in the Markdown format.
        * Respond in markdown format.
        """,
        parse_all_pages=True,
        ).load_data(main_filepath)
    logging.info(f"Number of pages for {main_filename}: {len(documents_main)}")

    # Concatenate pages
    all_pages_main = concatenate_pages(documents_main)

    # Remove markdown wrappers
    updated_all_pages_main = re.sub(r"(```markdown|``````markdown|```|``````)", "\n\n", all_pages_main)
    
    # Chunk into nodes
    document_main = Document(text=updated_all_pages_main)
    nodes_main_markdown = markdown_parser.get_nodes_from_documents([document_main])
    logging.info(f"Number of chunk nodes for {main_filename}: {len(nodes_main_markdown)}")

    # Consolidate nodes
    all_nodes.extend(nodes_main_markdown)
    logging.info(f"Total nodes consolidated after parse {main_filename}: {len(all_nodes)}")

    # Add new nodes to the existing index
    index_budget.insert_nodes(nodes_main_markdown)
    logging.info(f"Added nodes into vector store for: {main_filename}")

if j == len(main_filepaths) - 1:
    logging.info("All main files processed.")
else:
    logging.info(f"Processed {j+1} out of {len(main_filepaths)} main files.")



# ---------------------- Keyword index setup and populate --------------------- #
## BM25Retriever does not support dynamically adding of new nodes 

# Create text store index 'T1'
retriever_bm25_budget = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)

# Persist text store
if retriever_bm25_budget:
    logging.info(f"Added all nodes into text store")
    retriever_bm25_budget.persist("./budget2024_keyword")