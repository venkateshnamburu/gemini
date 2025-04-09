import streamlit as st
import boto3
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import io
import re

# --- CONFIG ---
GEMINI_API_KEY = "AIzaSyAwlJ-6BfIS3HeMPsjYU9Pc0A89lgykjuw"  # Replace with your actual API key
BUCKET_NAME = "heathrecords"
AWS_REGION = "us-east-1"

# --- SETUP ---
genai.configure(api_key=GEMINI_API_KEY)
s3 = boto3.client("s3", region_name=AWS_REGION)

# --- Extract PDF Text ---
def extract_text_from_pdf_stream(pdf_stream):
    doc = fitz.open("pdf", pdf_stream)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Ask Gemini ---
def ask_gemini_with_doc(text, question, filename="unknown_file"):
    prompt = f"""
You are a smart document understanding model.

Below is the extracted text from a document named: {filename}

{text}

Now, based on this content, answer the following user question:

"{question}"

Respond ONLY with a valid JSON object. Do not include explanations, markdown, or any extra text. Only output a JSON object with double-quoted keys and string values.

If a value is unavailable, use null. Ensure the response is parseable JSON.
"""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Clean Markdown or Noise from Output ---
def clean_json_text(text):
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
    return cleaned.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("📄💬 Ask Questions About PDF Files (From S3)")

# Load PDF files from S3
try:
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    pdf_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]
except Exception as e:
    st.error(f"Error accessing S3 bucket: {e}")
    st.stop()

selected_pdf = st.selectbox("Choose a PDF from S3 bucket:", pdf_files)

if selected_pdf:
    st.write(f"**Selected File:** `{selected_pdf}`")

    try:
        # Get PDF content
        obj_response = s3.get_object(Bucket=BUCKET_NAME, Key=selected_pdf)
        file_stream = obj_response["Body"].read()
        extracted_text = extract_text_from_pdf_stream(file_stream)

        # Input question
        question = st.text_input("❓ Ask a question about this document:")

        if question:
            with st.spinner("Thinking..."):
                raw_response = ask_gemini_with_doc(extracted_text, question, filename=selected_pdf)
                cleaned_response = clean_json_text(raw_response)

                try:
                    json_output = json.loads(cleaned_response)
                    st.success("✅ Parsed valid JSON response:")
                    st.json(json_output)
                except json.JSONDecodeError:
                    st.error("❌ Couldn't parse Gemini response into valid JSON.")
                    st.code(raw_response)
    except Exception as e:
        st.error(f"Error reading or processing PDF: {e}")
