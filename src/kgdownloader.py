import csv
from SPARQLWrapper import SPARQLWrapper, JSON

def download_ireland_kg_csv(
    output_file="ireland_kg_debug.csv",
    batch_size=10000,
    max_file_size_gb=1
):
    """
    Download Ireland-related triples from DBpedia in batches and save as CSV.
    CSV columns: subject, predicate, object.
    Fully debug-friendly with clear INFO messages.
    """
    print("[INFO] Starting Ireland KG CSV download...")
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    offset = 0
    total_bytes_written = 0
    max_bytes = max_file_size_gb * (1024 ** 3)  # Convert GB to bytes
    min_batch_bytes = 1024

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject", "predicate", "object"])  # CSV header
        print(f"[INFO] CSV file created: {output_file}")

        while total_bytes_written < max_bytes:
            print(f"[INFO] Fetching batch starting at OFFSET {offset} (batch size {batch_size})...")
            
            sparql.setQuery(f"""
            SELECT ?s ?p ?o WHERE {{
                {{
                    ?s ?p ?o .
                    ?s dbo:country dbr:Ireland .
                }}
                UNION
                {{
                    ?s ?p dbr:Ireland .
                }}
                UNION
                {{
                    dbr:Ireland ?p ?o .
                }}
            }} LIMIT {batch_size} OFFSET {offset}
            """)
            sparql.setReturnFormat(JSON)

            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            num_triples = len(bindings)
            print(f"[INFO] Retrieved {num_triples} triples in this batch.")

            if num_triples == 0:
                print("[DONE] No more results returned from SPARQL endpoint.")
                break

            batch_bytes = 0
            missing_o_count = 0

            for row in bindings:
                s = row.get("s", {}).get("value", "")
                p = row.get("p", {}).get("value", "")
                o = row.get("o", {}).get("value", "")
                
                if o == "":
                    missing_o_count += 1

                writer.writerow([s, p, o])
                batch_bytes += len(s) + len(p) + len(o)

            total_bytes_written += batch_bytes
            print(f"[INFO] Wrote {num_triples} triples, approx {batch_bytes / (1024 ** 2):.2f} MB")
            if missing_o_count > 0:
                print(f"[WARNING] {missing_o_count} triples had missing object values.")

            offset += batch_size
            csvfile.flush()
            print(f"[INFO] CSV file flushed. Total written so far: {total_bytes_written / (1024 ** 2):.2f} MB\n")

            if total_bytes_written >= max_bytes:
                print("[STOP] Reached max file size limit.")
                break

    print(f"[SUCCESS] Saved KG data to {output_file} ({total_bytes_written / (1024 ** 3):.2f} GB)")
    print("[INFO] Download completed.")

if __name__ == "__main__":
    download_ireland_kg_csv(
        output_file="ireland_kg_debug.csv",
        batch_size=20000,
        max_file_size_gb=1
    )
