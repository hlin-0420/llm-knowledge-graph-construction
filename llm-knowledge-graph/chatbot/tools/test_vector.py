from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()  # This loads environment variables from the .env file

# Verify the variables
print("NEO4J_URI:", os.getenv("NEO4J_URI"))
print("NEO4J_USERNAME:", os.getenv("NEO4J_USERNAME"))
# Note: Avoid printing passwords in production
# print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Initialize the Neo4j driver
driver = GraphDatabase.driver(uri, auth=(username, password))

# Define the Cypher query to check for the vector index
query = """
SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options
WHERE type = 'VECTOR'
RETURN name, entityType, labelsOrTypes, properties, options
"""

# Execute the query
with driver.session() as session:
    result = session.run(query)
    indexes = result.data()

# Close the driver
driver.close()

# Print all vector indexes
if indexes:
    print("📊 Vector indexes found in Neo4j:")
    for idx in indexes:
        print(idx)
else:
    print("❌ No vector indexes found.")

# Check if the 'vector' index exists specifically
vector_index = next((idx for idx in indexes if idx['name'] == 'vector'), None)

if vector_index:
    print("\n✅ Vector index 'vector' exists:")
    print(vector_index)
else:
    print("\n❌ Vector index 'vector' does not exist.")