import pdfplumber
import requests
import json
from sentence_transformers import SentenceTransformer, util

# Setting Gemini API key and URL
GEMINI_API_KEY = "AIzaSyDze59Xi148KrS_Xc_Rx_Sv7J9tNyi0hII"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

#Pdf processing and text extraction
def extract_pdf_text(file_path):
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append((i + 1, text))  # Store page number too
    return pages_text

# Matching user question with relevant section in PDF
def find_relevant_section(pages_text, query, top_k=1):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [text for _, text in pages_text]
    doc_embeddings = embedder.encode(texts, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    top_hits = similarities.topk(top_k) #Finds the top_k most similar PDF sections to the question.

    results = []
    for score, idx in zip(top_hits.values, top_hits.indices):
        page_num, text = pages_text[idx]
        results.append({
            "page": page_num,
            "score": float(score),
            "text": text
        })
    return results

# ========= CALL GEMINI API VIA HTTP =========
def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f" Error {response.status_code}: {response.text}"

#Explaining the relevant text using Gemini
def explain_text_with_gemini(text, question):
    prompt = f"""
A user asked: "{question}"

Here is an excerpt from a book that might help:

{text}

Please explain the relevant information clearly and thoroughly.
"""
    return call_gemini_api(prompt)

# Main function to process the PDF and answer the question
def ask_question_about_pdf(pdf_path, user_question):
    print(" Reading and processing PDF...")
    pages = extract_pdf_text(pdf_path)

    print(" Finding the most relevant section...")
    relevant = find_relevant_section(pages, user_question, top_k=1)[0]

    print(f"\n Relevant content found on page {relevant['page']}:\n")
    print(relevant['text'][:1000], "...")  # Truncate for display

    print("\n Gemini explanation:\n")
    explanation = explain_text_with_gemini(relevant['text'], user_question)
    print(explanation)

# User input and execution
if __name__ == "__main__":
    pdf_file_path = input(" Enter the path to the PDF file: ")
    if not pdf_file_path.endswith('.pdf'):
        print(" Please provide a valid PDF file.")
        exit(1)
    question = input(" Enter your question: ")
    ask_question_about_pdf(pdf_file_path, question)
