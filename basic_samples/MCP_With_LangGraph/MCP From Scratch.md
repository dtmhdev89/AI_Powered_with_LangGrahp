# MCP From Scratch

### Video

https://www.youtube.com/watch?v=CDjjaTALI68

### Scope

- Shop MCP tool connected with Cursor, Windsurf, Claude Desktop
- What are `context` and `tools`
- Build our own tool from scratch (for retrieving LangGraph docs)
- Convert it to an MCP server
- Connect it to Claude Desktop, Cursor, Windsurf

### Augmenting LLMs with context and tools

- Context → RAG (see: ‣)
- Tools → Agents (see:  https://langchain-ai.github.io/langgraph/tutorials/workflows/)

![image.png](MCP%20From%20Scratch%201ff6ca6fd6ef80aeb2dad10c4cfc95fa/image.png)

### Build a Tool from scratch

- Tool to retrieve LangGraph docs

```bash
! python3 -m venv .venv
! source .venv/bin/activate
! pip install langchain_community langchain-openai langchain-anthropic scikit-learn bs4 pandas pyarrow matplotlib lxml langgraph "mcp[cli]"
```

```python
import re, os
import tiktoken

from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import SKLearnVectorStore

def count_tokens(text, model="cl100k_base"):
    """
    Count the number of tokens in the text using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The tokenizer model to use (default: cl100k_base for GPT-4)
        
    Returns:
        int: Number of tokens in the text
    """
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    
    # Target the main article content for LangGraph documentation 
    main_content = soup.find("article", class_="md-content__inner")
    
    # If found, use that, otherwise fall back to the whole document
    content = main_content.get_text() if main_content else soup.text
    
    # Clean up whitespace
    content = re.sub(r"\n\n+", "\n\n", content).strip()
    
    return content

def load_langgraph_docs():
    """
    Load LangGraph documentation from the official website.
    
    This function:
    1. Uses RecursiveUrlLoader to fetch pages from the LangGraph website
    2. Counts the total documents and tokens loaded
    
    Returns:
        list: A list of Document objects containing the loaded content
        list: A list of tokens per document
    """
    print("Loading LangGraph documentation...")

    # Load the documentation 
    urls = ["https://langchain-ai.github.io/langgraph/concepts/",
     "https://langchain-ai.github.io/langgraph/how-tos/",
     "https://langchain-ai.github.io/langgraph/tutorials/workflows/",  
     "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
     "https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/",
    ] 

    docs = []
    for url in urls:

        loader = RecursiveUrlLoader(
            url,
            max_depth=5,
            extractor=bs4_extractor,
        )

        # Load documents using lazy loading (memory efficient)
        docs_lazy = loader.lazy_load()

        # Load documents and track URLs
        for d in docs_lazy:
            docs.append(d)

    print(f"Loaded {len(docs)} documents from LangGraph documentation.")
    print("\nLoaded URLs:")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.metadata.get('source', 'Unknown URL')}")
    
    # Count total tokens in documents
    total_tokens = 0
    tokens_per_doc = []
    for doc in docs:
        total_tokens += count_tokens(doc.page_content)
        tokens_per_doc.append(count_tokens(doc.page_content))
    print(f"Total tokens in loaded documents: {total_tokens}")
    
    return docs, tokens_per_doc

def save_llms_full(documents):
    """ Save the documents to a file """

    # Open the output file
    output_filename = "llms_full.txt"

    with open(output_filename, "w") as f:
        # Write each document
        for i, doc in enumerate(documents):
            # Get the source (URL) from metadata
            source = doc.metadata.get('source', 'Unknown URL')
            
            # Write the document with proper formatting
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {source}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")

    print(f"Documents concatenated into {output_filename}")

def split_documents(documents):
    """
    Split documents into smaller chunks for improved retrieval.
    
    This function:
    1. Uses RecursiveCharacterTextSplitter with tiktoken to create semantically meaningful chunks
    2. Ensures chunks are appropriately sized for embedding and retrieval
    3. Counts the resulting chunks and their total tokens
    
    Args:
        documents (list): List of Document objects to split
        
    Returns:
        list: A list of split Document objects
    """
    print("Splitting documents...")
    
    # Initialize text splitter using tiktoken for accurate token counting
    # chunk_size=8,000 creates relatively large chunks for comprehensive context
    # chunk_overlap=500 ensures continuity between chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,  
        chunk_overlap=500  
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Created {len(split_docs)} chunks from documents.")
    
    # Count total tokens in split documents
    total_tokens = 0
    for doc in split_docs:
        total_tokens += count_tokens(doc.page_content)
    
    print(f"Total tokens in split documents: {total_tokens}")
    
    return split_docs

def create_vectorstore(splits):
    """
    Create a vector store from document chunks using SKLearnVectorStore.
    
    This function:
    1. Initializes an embedding model to convert text into vector representations
    2. Creates a vector store from the document chunks
    
    Args:
        splits (list): List of split Document objects to embed
        
    Returns:
        SKLearnVectorStore: A vector store containing the embedded documents
    """
    print("Creating SKLearnVectorStore...")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create vector store from documents using SKLearn
    persist_path = os.getcwd()+"/sklearn_vectorstore.parquet"
    vectorstore = SKLearnVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_path=persist_path   ,
        serializer="parquet",
    )
    print("SKLearnVectorStore created successfully.")
    
    vectorstore.persist()
    print("SKLearnVectorStore was persisted to", persist_path)

    return vectorstore
```

```python
# Load the documents
documents, tokens_per_doc = load_langgraph_docs()

# Save the documents to a file
save_llms_full(documents)

# Split the documents
split_docs = split_documents(documents)

# Create the vector store
vectorstore = create_vectorstore(split_docs)
```

```python
# Create retriever to get relevant documents (k=3 means return top 3 matches)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
# Get relevant documents for the query
query = "What is LangGraph?"    
relevant_docs = retriever.invoke(query)
print(f"Retrieved {len(relevant_docs)} relevant documents")

for d in relevant_docs:
    print(d.metadata['source'])
    print(d.page_content[0:500])
    print("\n--------------------------------\n")
```

```python
from langchain_core.tools import tool

@tool
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"), 
    persist_path=os.getcwd()+"/sklearn_vectorstore.parquet", 
    serializer="parquet").as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context
```

We can [bind](https://blog.langchain.dev/tool-calling-with-langchain/) this tool to LLM

```python
llm = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0)
augmented_llm = llm.bind_tools([langgraph_query_tool])

instructions = """You are a helpful assistant that can answer questions about the LangGraph documentation. 
Use the langgraph_query_tool for any questions about the documentation.
If you don't know the answer, say "I don't know."""

messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": "What is LangGraph?"}
]

message = augmented_llm.invoke(messages)
message.pretty_print()
```

### MCP

- Now, we have a tool `langgraph_query_tool` and we can connect it to our chat model
- But how can I connect this with Cursor, Claude Code/Desktop, or Windsurf?

![image.png](MCP%20From%20Scratch%201ff6ca6fd6ef80aeb2dad10c4cfc95fa/image%201.png)

- MCP gives us a standard protocol for doing this, analogous to `bind_tools`
    - Both provide interfaces for AI models to interact with external tools and data sources
    - Both support the concept of exposing "tools" that models can invoke

### MCP Protocol

https://modelcontextprotocol.io/introduction

- Host/Client
    - Application (IDE, Claude Code)
- Server
    - [Resources](https://modelcontextprotocol.io/docs/concepts/resources) → Docs
    - [Tools](https://modelcontextprotocol.io/docs/concepts/tools) → Tools, as above
    - [Prompts](https://modelcontextprotocol.io/docs/concepts/prompts)
- [Communication](https://modelcontextprotocol.io/docs/concepts/transports)
    - [Stdio](https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio)
    - [SSE](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse)

![image.png](MCP%20From%20Scratch%201ff6ca6fd6ef80aeb2dad10c4cfc95fa/image%202.png)

Define a server, `langgraph-mcp.py`

```python
#!/Users/rlm/Desktop/Code/vibe-code-benchmark/.venv/bin/python

from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

# Define common path to the repo locally
PATH = "/Users/rlm/Desktop/Code/vibe-code-benchmark/"

# Create an MCP server
mcp = FastMCP("LangGraph-Docs-MCP-Server")

# Add a tool to query the LangGraph documentation
@mcp.tool()
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.
    
    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"), 
        persist_path=PATH + "sklearn_vectorstore.parquet", 
        serializer="parquet").as_retriever(search_kwargs={"k": 3}
        )

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join([f"==DOCUMENT {i+1}==\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return formatted_context

# The @mcp.resource() decorator is meant to map a URI pattern to a function that provides the resource content
@mcp.resource("docs://langgraph/full")
def get_all_langgraph_docs() -> str:
    """
    Get all the LangGraph documentation. Returns the contents of the file llms_full.txt,
    which contains a curated set of LangGraph documentation (~300k tokens). This is useful
    for a comprehensive response to questions about LangGraph.

    Args: None

    Returns:
        str: The contents of the LangGraph documentation
    """

    # Local path to the LangGraph documentation
    doc_path = PATH + "llms_full.txt"
    try:
        with open(doc_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
```

Test this with inspector

https://modelcontextprotocol.io/docs/tools/inspector

```python
npx @modelcontextprotocol/inspector
```

- COMMAND: `/Users/rlm/Desktop/Code/test-mcp/.venv/bin/python`
- ARGUMENTS: `/Users/rlm/Desktop/Code/test-mcp/langgraph-mcp.py`

### Configure Hosts

```python
{
    "mcpServers": {
        "langgraph-mcp": {
            "command": "/Users/rlm/Desktop/Code/test-mcp/.venv/bin/python",
            "args": [
                "/Users/rlm/Desktop/Code/test-mcp/langgraph-mcp.py"
            ],
            "env": {
                "OPENAI_API_KEY": "<your-openai-api-key>"
            }
        }
    }
}
```

**Cursor** 
`~/.cursor/mcp.json` 

**Windsurf**
`~/.codeium/windsurf/mcp_config.json`

 **Claude Desktop**
`~/Library/Application\ Support/Claude/claude_desktop_config.json`

**Claude Code**
`~/.claude.json`