import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def judge_faithfulness(question, context, answer, max_retries=3):
    prompt = f"""You are a strict fact-checker. Determine if the ANSWER is fully supported by the CONTEXT, with no invented or unsupported claims.

Context: {context}

Question: {question}

Answer: {answer}

Respond in exactly this format:
VERDICT: [FAITHFUL or UNFAITHFUL]
REASON: [one short sentence explaining why]"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            print(f"  Judge call failed ({e}), retrying in {wait_time}s...")
            time.sleep(wait_time)

    return "VERDICT: ERROR\nREASON: Failed after retries"
if __name__ == "__main__":
    result = judge_faithfulness(
        question="Who created Python?",
        context="Python was created by Guido van Rossum and first released in 1991.",
        answer="Guido van Rossum created Python while working at Google."
    )
    print(result)