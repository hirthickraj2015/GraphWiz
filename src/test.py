from neo4j import GraphDatabase

uri = "neo4j+s://d3c3a325.databases.neo4j.io"
user = "neo4j"
pwd = "uZHvPLzNNUlqVewtkhKXSpME3TkSJSZy2LOx70d5puc"

driver = GraphDatabase.driver(uri, auth=(user, pwd))
with driver.session() as session:
    result = session.run("RETURN 1 AS test")
    print(result.single())
