import pandas as pd
from neo4j import GraphDatabase
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
import time
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# ---------------- CONFIG ----------------
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

DATASET_PATH = "datasets/hotels.csv"
MAX_ROWS_TO_PROCESS = 200
# ----------------------------------------

# -------- GLOBAL TAG CACHE (massive speed improvement) -------
TAG_CACHE = {}


# ---------------- HELPER: NORMALIZE HOTEL RATING ----------------
def normalize_rating(rating):
    rating = str(rating).strip().lower()

    if "one" in rating:
        return 1.0
    if "two" in rating:
        return 2.0
    if "three" in rating:
        return 3.0
    if "four" in rating:
        return 4.0
    if "five" in rating:
        return 5.0

    # Cases like 'All', 'NA', '', unknown
    return 0.0
# ----------------------------------------------------------------


# ---------------- LOCAL LLM INITIALIZATION ----------------
def load_local_llm():
    print("🔄 Loading local free LLM (Flan-T5-Base)...")

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    print("✅ Local LLM Loaded!")
    return tokenizer, model


# ---------------- LOCAL TAG EXTRACTION ----------------
def extract_semantic_tags_local(tokenizer, model, hotel_name, description):

    # 🔥 Truncate very long descriptions (T5 limit = 512 tokens)
    if len(description) > 1000:
        description = description[:1000]

    system_prompt = (
        "Extract semantic tags describing the vibe of a hotel. "
        "Return ONLY JSON like: {\"tags\": [\"romantic\", \"luxury\"]}."
    )

    prompt = f"{system_prompt}\nHotel: {hotel_name}\nDescription: {description}"

    # ---- TAG CACHE CHECK ----
    cache_key = prompt[:300]  # reduce memory
    if cache_key in TAG_CACHE:
        return TAG_CACHE[cache_key]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON
        start = result.find("{")
        end = result.rfind("}") + 1

        if start != -1 and end != -1:
            json_text = result[start:end]
            parsed = json.loads(json_text)
            tags = parsed.get("tags", [])

            TAG_CACHE[cache_key] = tags  # save to cache
            return tags

    except Exception as e:
        print(f"⚠️ Tag extraction error for {hotel_name}: {e}")

    TAG_CACHE[cache_key] = []
    return []
# ----------------------------------------------------


# ---------------- NEO4J DRIVER ----------------
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Neo4j connected")
        return driver
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return None


# ---------------- INGESTION LOGIC ----------------
def process_and_ingest_row(session, row, tokenizer, model):
    try:
        hotel_code = row.get("HotelCode")
        hotel_name = row.get("HotelName", "Unknown")
        description = row.get("Description", "")
        city_name = row.get("cityName", "")
        rating_raw = row.get("HotelRating", "0")
        address = row.get("Address", "")

        # ---- Skip invalid rows ----
        if not hotel_code or str(hotel_code).strip() in ("", "nan", "None"):
            print(f"⚠️ Skipping row (missing HotelCode) → {hotel_name}")
            return

        hotel_code = str(hotel_code).strip()

        # ---- Fix rating ----
        rating = normalize_rating(rating_raw)

        # ---- Extract tags ----
        semantic_tags = extract_semantic_tags_local(tokenizer, model, hotel_name, description)

        hotel_props = {
            "HotelCode": hotel_code,
            "name": hotel_name,
            "rating": rating,
            "address": address,
            "description": description
        }

        cypher_query = """
        MERGE (h:Hotel {HotelCode: $hotel.HotelCode})
        ON CREATE SET h = $hotel
        ON MATCH SET h.name = $hotel.name,
                      h.rating = $hotel.rating,
                      h.address = $hotel.address,
                      h.description = $hotel.description

        MERGE (c:City {name: $city_name})
        MERGE (h)-[:LOCATED_IN]->(c)

        WITH h, $tags AS tags
        UNWIND tags AS tname
            MERGE (t:Tag {name: tname})
            MERGE (h)-[:HAS_TAG]->(t)
        """

        session.run(
            cypher_query,
            hotel=hotel_props,
            city_name=city_name,
            tags=semantic_tags
        )

        print(f"  ✅ Inserted: {hotel_name} ({hotel_code})")

    except Exception as e:
        print(f"❌ Neo4j ingestion error: {e}")


# ---------------- MAIN ----------------
def main():
    tokenizer, model = load_local_llm()

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    driver = get_neo4j_driver()
    if not driver:
        return

    print(f"\n🚀 Ingesting {MAX_ROWS_TO_PROCESS} rows...\n")

    chunk_iter = pd.read_csv(DATASET_PATH, chunksize=10, encoding="latin-1")

    rows_processed = 0

    for chunk in chunk_iter:

        # Clean columns
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.astype(str).fillna("")

        with driver.session() as session:
            for _, row in chunk.iterrows():
                if rows_processed >= MAX_ROWS_TO_PROCESS:
                    break

                process_and_ingest_row(session, row, tokenizer, model)
                rows_processed += 1

        if rows_processed >= MAX_ROWS_TO_PROCESS:
            break

    driver.close()

    print("\n✅ --- Ingestion complete! ---")
    print(f"Inserted {rows_processed} hotels into Neo4j.")


if __name__ == "__main__":
    main()
