
def split_file_with_tab(path):
    list1, list2 = [], []
    with open(path, encoding="utf-8")as f:
        while True:
            l = f.readline()
            if l:
                # Tabについて分割
                s = l.split('\t')
                # 語りと応答について分割
                list1.append(s[0] + '\n')
                # 謎に改行コードが入ったので消しておく
                list2.append(s[1])
            else:
                break
    with open('speak.txt', mode='w', encoding='utf-8') as s:
        s.writelines(list1)
    with open('res.txt', mode='w', encoding='utf-8') as s:
        s.writelines(list2)


def split_word(word_list):
    return word_list.split(' ')
