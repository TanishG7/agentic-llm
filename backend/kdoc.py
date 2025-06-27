import json
from neo4j import GraphDatabase

# --------------- Neo4j Credentials ---------------
uri = "neo4j+s://77c24b62.databases.neo4j.io"  # Default Neo4j connection
user = "neo4j"

password = "SKqxhGMkECZBQnMMXCugw49kF93R3cC4UNbbl9Huspw"  # Replace with your actual password

# --------------- Neo4j Connection ---------------
driver = GraphDatabase.driver(uri, auth=(user, password))

# --------------- Function to Create Node ---------------
def create_node(tx, labels, properties):
    # Dynamically handle labels
    label_string = ":".join(labels)
    query = f"""
    CREATE (n:{label_string})
    SET n += $props
    """
    tx.run(query, props=properties)

# --------------- Batch Processing Setup ---------------
batch_size = 1000
batch = []

# --------------- Load JSON & Process ---------------
with open('kdoc_export.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with driver.session() as session:
    for record in data:
        labels = record.get("labels", ["KDOC"])  # Fallback to 'KDOC' if missing
        properties = record.get("properties", {})

        batch.append((labels, properties))

        if len(batch) == batch_size:
            for labels, props in batch:
                session.execute_write(create_node, labels, props)

            batch.clear()

    # Process remaining records
    for labels, props in batch:
        session.execute_write(create_node, labels, props)


driver.close()

print("✅ Import Complete")
