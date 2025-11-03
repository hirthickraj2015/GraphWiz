from neo4j import GraphDatabase
import pandas as pd


NEO4J_URI = "neo4j+s://d3c3a325.databases.neo4j.io"  # Replace with your Aura URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "uZHvPLzNNUlqVewtkhKXSpME3TkSJSZy2LOx70d5puc"
CSV_FILE = "../dataset/ireland_kg.csv"  
BATCH_SIZE = 1000                        


# Step 1: Load CSV and clean
print("[INFO] Loading CSV file...")
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=["subject", "object"])
df["predicate"] = df["predicate"].fillna("relatedTo")
print(f"[INFO] Total triples after cleaning: {len(df)}")


# Step 2: Connect to Neo4j Aura
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print("[INFO] Connected to Neo4j Aura Cloud!")


# Step 3: Determine starting index for resuming
with driver.session() as session:
    # Count existing relationships in the DB
    existing_rels = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS count").single()["count"]
    start_index = existing_rels  # Resume from next triple
    print(f"[INFO] Resuming upload from triple index: {start_index}")

# Step 4: Upload in batches
def upload_batch(batch):
    with driver.session() as session:
        records = [
            {"s": row["subject"], "p": row["predicate"], "o": row["object"]}
            for _, row in batch.iterrows()
        ]
        session.run("""
        UNWIND $records AS row
        MERGE (subject:Entity {uri: row.s})
        MERGE (object:Entity {uri: row.o})
        MERGE (subject)-[:REL {type: row.p}]->(object)
        """, records=records)


print("[INFO] Uploading triples to Neo4j in batches...")

for i in range(start_index, len(df), BATCH_SIZE):
    batch = df.iloc[i:i+BATCH_SIZE]
    upload_batch(batch)
    print(f"[INFO] Uploaded triples {i} to {i + len(batch)}")


# Step 5: Verify upload
with driver.session() as session:
    node_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
    rel_count = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS count").single()["count"]
    print(f"[SUCCESS] Upload complete. Nodes: {node_count}, Relationships: {rel_count}")


driver.close()
print("[INFO] Neo4j connection closed.")