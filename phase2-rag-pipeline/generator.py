import os
from dotenv import load_dotenv
from google import genai
from retriever import retrieve


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(question, context):
    prompt = f"""Answer the question using ONLY the information in the context below.
If the context does not contain the answer, say "I don't have enough information to answer this."

Context: {context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

question = "Who created Python?"
doc, score = retrieve(question)
answer = generate_answer(question, doc)

print("Retrieved context:", doc)
print("Answer:", answer)