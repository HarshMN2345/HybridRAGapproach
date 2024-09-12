import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables from .env (if needed for API keys)
load_dotenv()

# Set up the parser with LlamaParse
parser = LlamaParse(result_type="text")  # Extract plain text

# Use SimpleDirectoryReader to parse the PDF file
file_extractor = {".pdf": parser}
pdf_path = 'Test.pdf'  # Replace with your actual PDF path
documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()

# Combine all extracted text from documents
extracted_text = "\n".join([doc.text for doc in documents])

# Save the extracted text into extracted_text.txt
output_path = 'extracted_text.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)

print(f"Text and tables have been saved to {output_path}")
