import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from janome.tokenizer import Tokenizer

j_t = Tokenizer()

def tokenizer(sentence):
    return [tok for tok in j_t.tokenize(sentence, wakati=True)]

def eval_score(df):
    # answerの中はリストになっていてほしい
    ans_list = df["answer"]
    pred_list = df["predict"]

    # 一致率の計算
    cnt = 0
    for i in range(len(pred_list)):
        predicted_sentence = pred_list[i]
        for answer in ans_list[i]:
            if answer == predicted_sentence:
                cnt += 1
    percentage = (cnt/len(ans_list)) * 100

    # 種類数
    express_kinds = []
    for predict in pred_list:
        if predict not in express_kinds:
            express_kinds.append(predict)
    express_kind = len(express_kinds)

    # bleu
    bleu_scores, answer= [], []
    for i in range(len(pred_list)):
        pred_sentence = pred_list[i]
        predict = tokenizer(pred_sentence)
        ans = ans_list[i]
        for ans_str in ans:
            answer.append(tokenizer(ans_str))
        # print(answer)
        # print(predict)
        bleu_scores.append(sentence_bleu(answer, predict))

        answer = []

    bleu_score = sum(bleu_scores)/len(bleu_scores)

    return percentage, express_kind, bleu_score