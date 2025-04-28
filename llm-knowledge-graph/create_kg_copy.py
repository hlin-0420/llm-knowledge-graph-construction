import os

from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from pyvis.network import Network
import webbrowser
from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from dotenv import load_dotenv
load_dotenv()

DOCS_PATH = "llm-knowledge-graph/data/course/html"

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-3.5-turbo"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes = ["Technology", "Concept", "Skill", "Event", "Person", "Object"],
    allowed_relationships=["USES", "HAS", "IS", "AT", "KNOWS"],
    )

# Load and split the documents
loader = DirectoryLoader(DOCS_PATH, glob="**/*.htm", loader_cls=BSHTMLLoader)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:

    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata.get('page', '0')}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Fetch all nodes and relationships
result = graph.query("""
    MATCH (n)-[r]->(m)
    RETURN n, r, m
""")

if result:
    print("Available keys in the first record:", result[0].keys())
    print("Example record values:", result[0])
else:
    print("No results found in the graph query.")

# Create a Pyvis Network
net = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add nodes and edges to the Pyvis network
added_nodes = set()

for idx, record in enumerate(result):
    node_1 = record['n']
    node_2 = record['m']
    relationship = record['r']

    # Safe fallback to internal Neo4j node ID if 'id' is missing
    node_1_id = node_1.get('id') or f"node1_{idx}"
    node_1_type = node_1.get('type', 'Node')

    node_2_id = node_2.get('id') or f"node2_{idx}"
    node_2_type = node_2.get('type', 'Node')

    # Add node_1 if not already added
    if node_1_id not in added_nodes:
        net.add_node(node_1_id, label=f"{node_1_id} ({node_1_type})", title=str(node_1))
        added_nodes.add(node_1_id)

    # Add node_2 if not already added
    if node_2_id not in added_nodes:
        net.add_node(node_2_id, label=f"{node_2_id} ({node_2_type})", title=str(node_2))
        added_nodes.add(node_2_id)

    net.add_edge(node_1_id, node_2_id)

# Save and open HTML visualisation
output_dir = "graph_visualisation.html"
net.write_html(output_dir)

webbrowser.open("file://" + os.path.abspath(output_dir))
print(f"Interactive graph saved as {output_dir}")