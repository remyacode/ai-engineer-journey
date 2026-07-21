import time
from eval_dataset import EVAL_SET
from retriever import retrieve
from generator import generate_answer

def run_evaluation():
    results = []
    
    for i, item in enumerate(EVAL_SET):
        question = item["question"]
        expected = item["expected"]
        
        doc, score = retrieve(question)
        answer = generate_answer(question, doc)
        
        passed = expected.lower() in answer.lower()
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "retrieved_doc": doc,
            "passed": passed
        })
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(EVAL_SET)}, pausing to respect rate limit...")
            time.sleep(60)
    
    return results

def print_results(results):
    passed_count = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed_count += 1
        print(f"[{status}] Q: {r['question']}")
        print(f"       Expected: {r['expected']}")
        print(f"       Got: {r['answer']}")
        print()
    
    accuracy = passed_count / len(results) * 100
    print(f"Accuracy: {passed_count}/{len(results)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    results = run_evaluation()
    print_results(results)
    with open("eval_results.txt", "w") as f:
        for r in results:
            f.write(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['question']}\n")
            f.write(f"  Expected: {r['expected']}\n")
            f.write(f"  Got: {r['answer']}\n\n")
        passed_count = sum(1 for r in results if r['passed'])
        f.write(f"Accuracy: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)\n")