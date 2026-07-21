import time
from retriever import retrieve
from faithfulness_judge import judge_faithfulness

# Reusing answers already generated and saved in eval_results.txt,
# to avoid spending quota re-generating them.
SAVED_RESULTS = [
    {"question": "Who created Python?", "answer": "Guido van Rossum"},
    {"question": "What year was the transformer architecture introduced?", "answer": "The transformer architecture was introduced in 2017."},
    {"question": "Who created Git?", "answer": "Linus Torvalds"},
    {"question": "What does GPT stand for?", "answer": "Generative Pre-trained Transformer."},
    {"question": "What year was the first iPhone released?", "answer": "2007"},
    {"question": "Who developed PyTorch?", "answer": "PyTorch was developed primarily by Facebook's AI Research lab (FAIR)."},
    {"question": "Who coined the term artificial intelligence?", "answer": "John McCarthy"},
    {"question": "What year was Kubernetes released as open source?", "answer": "2014"},
    {"question": "Who published the RSA encryption algorithm?", "answer": "Rivest, Shamir, and Adleman."},
    {"question": "Who created React?", "answer": "Jordan Walke"},
]

def run_faithfulness_eval():
    results = []
    for i, item in enumerate(SAVED_RESULTS):
        question = item["question"]
        answer = item["answer"]
        context, _ = retrieve(question)

        verdict_text = judge_faithfulness(question, context, answer)
        results.append({
            "question": question,
            "answer": answer,
            "context": context,
            "verdict": verdict_text
        })

        if (i + 1) % 5 == 0 and (i + 1) < len(SAVED_RESULTS):
            print(f"Judged {i + 1}/{len(SAVED_RESULTS)}, pausing for rate limit...")
            time.sleep(60)

    return results

def save_results(results, filename="faithfulness_results.txt"):
    with open(filename, "w") as f:
        faithful_count = 0
        for r in results:
            f.write(f"Q: {r['question']}\n")
            f.write(f"A: {r['answer']}\n")
            f.write(f"{r['verdict']}\n\n")
            if "VERDICT: FAITHFUL" in r["verdict"]:
                faithful_count += 1
        f.write(f"Faithfulness rate: {faithful_count}/{len(results)} ({faithful_count/len(results)*100:.1f}%)\n")
    print(f"Saved to {filename}")

if __name__ == "__main__":
    results = run_faithfulness_eval()
    save_results(results)