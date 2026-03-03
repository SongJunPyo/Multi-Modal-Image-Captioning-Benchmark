

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
import json


def load_csv(path):
    """Returns dict: {filename: caption}"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(df["filename"], df["caption"]))


def compute_bleu_4(ref_caps, gen_caps):
    """
    BLEU-4: corpus-level
    """
    smoothie = SmoothingFunction().method4

    references = []
    hypotheses = []

    for f in gen_caps:   # loop through filenames
        references.append([ref_caps[f].split()])   # list of list of tokens
        hypotheses.append(gen_caps[f].split())     # list of tokens

    return corpus_bleu(references, hypotheses, smoothing_function=smoothie)


def compute_meteor(ref_caps, gen_caps):
    """
    METEOR: sentence-level averaged
    """
    scores = []
    for f in gen_caps:
        ref_tokens = ref_caps[f].split()
        hyp_tokens = gen_caps[f].split()
        scores.append(meteor_score([ref_tokens], hyp_tokens))
    return sum(scores) / len(scores)



def compute_cider_d(ref_caps, gen_caps):
    """
    CIDEr-D using pycocoevalcap implementation.
    Need dict:
      refs = {id: [caption]}
      hyps = {id: [caption]}
    """
    cider = Cider()

    refs = {i: [ref_caps[f]] for i, f in enumerate(gen_caps)}
    hyps = {i: [gen_caps[f]] for i, f in enumerate(gen_caps)}

    score, _ = cider.compute_score(refs, hyps)
    return score


def main():
   
    reference_csv = "/home/jpsong/imgProcessing_Teamproject/student_data/results/reference.csv" # reference 캡션 파일 (학습데이터에 미포함)
    generated_csv = "/home/jpsong/imgProcessing_Teamproject/student_data/results/generated.csv" # VLM 학습 후 테스트 이미지 (3,000장)에 대한 생성된 캡션 
    output_file = "/home/jpsong/imgProcessing_Teamproject/student_data/results/metrics.txt" # VLM에 대한 성능지표 출력 파일

    ref_caps = load_csv(reference_csv)
    gen_caps = load_csv(generated_csv)

    # Compute metrics
    bleu4 = compute_bleu_4(ref_caps, gen_caps)
    meteor = compute_meteor(ref_caps, gen_caps)
    cider_d = compute_cider_d(ref_caps, gen_caps)

    # Write to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"BLEU-4: {bleu4:.4f}\n")
        f.write(f"METEOR: {meteor:.4f}\n")
        f.write(f"CIDEr-D: {cider_d:.4f}\n")

    print("Metrics saved to:", output_file)


if __name__ == "__main__":
    main()
