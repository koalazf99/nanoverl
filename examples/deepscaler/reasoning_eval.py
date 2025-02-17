"""
Usage:
python -m sglang_router.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --port 30000 --dp-size 8
python reasoning_eval.py \
    --data-path nanoverl/aime \
    --parallel 256 \
    --num-tries 16 \
    --question-key problem
"""
import argparse
import json
import time

from datasets import load_dataset
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text

@sgl.function
def reasoning_gen(s, question: str):
    s += sgl.user(
        question
        + " Please reason step by step, and put your final answer within \\boxed{}."
    )
    s += sgl.assistant(
        sgl.gen(
            "answer",
        )
    )


def convert_dataset(path: str, question_key: str, answer_key: str, num_tries: int):
    raw_dataset = load_dataset(path)
    questions = []
    answers = []
    for data in raw_dataset["test"]:
        question = data[question_key]
        answer = data[answer_key]
        for _ in range(num_tries):
            questions.append({"question": question})
            answers.append({"answer": answer})
    return questions, answers


def main(args):
    # Select backend
    sgl.set_default_backend(select_sglang_backend(args))

    # Get dataset
    questions, answers = convert_dataset(
        args.data_path, args.question_key, args.answer_key, args.num_tries
    )

    # Run requests
    tic = time.time()
    states = reasoning_gen.run_batch(
        questions,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=0.6,
        max_new_tokens=32768,
        top_p=0.95,
    )
    latency = time.time() - tic

    # Extract answers
    # Calculate Pass@K, since we have multiple tries
    problem_group = dict()
    for i, state in enumerate(states):
        try:
            pred_answer = parse(
                state["answer"], 
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=True,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # turn number to string
            if isinstance(answers[i]["answer"], (int, float)):
                gt_answer = str(answers[i]["answer"])
            else:
                gt_answer = parse(answers[i]["answer"])
            
            correct = 1 if verify(pred_answer, gt_answer) else 0

            question = questions[i]["question"]
            if question not in problem_group:
                problem_group[question] = []
            if len(pred_answer) > 0:
                problem_group[question].append((correct, pred_answer[0]))
            else:
                problem_group[question].append((correct, None))
        except Exception as e:
            print(pred_answer, gt_answer)
            print(f"Error extracting answer: {e}")
            pass

    # Calculate Pass@1
    pass_1 = 0
    for question, results in problem_group.items():
        pass_1 += sum([1 for crt, _ in results if crt == 1])
    pass_1 /= len(states)
    print(f"Pass@1: {pass_1}")

    # Calculate Cons@K (Majority Vote)
    from collections import Counter
    cons_k = 0
    if args.num_tries > 1:
        for question, results in problem_group.items():
            # if most common answer is correct, then it is correct
            # print(Counter(results).most_common(1)[0][0][0])
            if Counter(results).most_common(1)[0][0][0] == 1:
                cons_k += 1
        cons_k /= len(problem_group)
        print(f"Cons@{args.num_tries}: {cons_k}")

    # Calculate output throughput
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency
    print(f"Output throughput: {output_throughput} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": args.data_path,
            "backend": args.backend,
            "latency": round(latency, 3),
            "pass_1": round(pass_1, 3),
            "cons_k": round(cons_k, 3),
            "num_requests": len(questions),
            "other": {
                "num_questions": len(questions),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="nanoverl/aime")
    parser.add_argument("--question-key", type=str, default="problem")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--num-tries", type=int, default=16)
    add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    main(args)