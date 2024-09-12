import os
from dotenv import load_dotenv
from transformers import pipeline
from neo4j import GraphDatabase
import spacy

# Initialize the NER pipeline from Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize spaCy for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Neo4j Database connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def clean_entity(word):
    """Merge tokenized words like ##word back into the full word."""
    return word.replace("##", "")

def extract_entities_transformers(text):
    """Extract named entities from text using Hugging Face Transformers."""
    entities = ner_pipeline(text)
    formatted_entities = [(clean_entity(ent['word']), ent['entity']) for ent in entities]
    return list(set(formatted_entities))  # Remove duplicate entities

def extract_relationships(text, entities):
    """Extract relationships between entities from the text using dependency parsing."""
    relationships = []
    doc = nlp(text)

    for sentence in doc.sents:
        sentence_entities = [ent for ent in entities if ent[0] in sentence.text]
        if len(sentence_entities) > 1:
            for ent1 in sentence_entities:
                for ent2 in sentence_entities:
                    if ent1 != ent2:
                        # Extract a verb or preposition between two entities (relation indicator)
                        for token in sentence:
                            if token.dep_ in ('ROOT', 'prep', 'agent') and ent1[0] in token.text and ent2[0] in token.text:
                                relationships.append((ent1, token.lemma_, ent2))
    return relationships

def create_neo4j_nodes(tx, entities):
    """Create nodes in Neo4j."""
    for entity, label in entities:
        query = "MERGE (e:Entity {name: $entity, label: $label})"
        tx.run(query, entity=entity, label=label)

def create_neo4j_relationships(tx, relationships):
    """Create specific relationships between nodes in Neo4j."""
    for (entity1, relation, entity2) in relationships:
        # Use a dynamic relationship type by formatting it directly into the query
        query = (
            "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
            "MERGE (e1)-[r:RELATED_TO {type: $relation}]->(e2)"
        )
        tx.run(query, entity1=entity1[0], entity2=entity2[0], relation=relation.upper().replace(" ", "_"))

def process_text_and_build_graph(text):
    """Process the text and add it to the knowledge graph."""
    entities = extract_entities_transformers(text)
    relationships = extract_relationships(text, entities)
    
    # Add entities and relationships to Neo4j
    with driver.session() as session:
        session.execute_write(create_neo4j_nodes, entities)
        session.execute_write(create_neo4j_relationships, relationships)
    
    print("Entities and relationships added to Neo4j Knowledge Graph.")

# Load text extracted from the PDF
try:
    with open('extracted_text.txt', 'r', encoding='utf-8') as file:
        pdf_text = file.read()
except UnicodeDecodeError as e:
    print(f"Error reading file: {e}")
    # Handle the error or reprocess the file with the correct encoding

# Process the text and build the Knowledge Graph
process_text_and_build_graph(pdf_text)

# Close the Neo4j driver connection
driver.close()
