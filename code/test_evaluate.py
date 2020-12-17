from evaluate import eval_score
import pandas as pd

df = pd.read_csv("../csv/result_Seq2seq.csv")

print(df[df.duplicated(subset = "input")])

df = df.groupby(["input", "predict"], as_index = False).agg(
    {"answer": list})

print(df.columns)

percentage, kinds, bleu = eval_score(df)
print(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
with open("test.txt", mode="r+", encoding="utf-8") as f:
    f.write(f"一致率: {percentage}, 種類数: {kinds}, BLEU: {bleu}")
