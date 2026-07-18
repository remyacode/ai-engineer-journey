import os
import time
from dotenv import load_dotenv
from google import genai
from retriever import retrieve


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(question, context, max_retries=3):
    prompt = f"""Answer the question using ONLY the information in the context below.
If the context does not contain the answer, say "I don't have enough information to answer this."

Context: {context}

Question: {question}

Answer:"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            print(f"  API call failed ({e}), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    return "ERROR: Failed to get response after retries"