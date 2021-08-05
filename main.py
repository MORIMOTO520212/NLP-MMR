import MeCab, math
import numpy as np

document = [] # 単語レベルのデータを格納する変数
tf_lst = [] # TF（頻出度）を格納する変数
idf_lst = [] # IDF（希少度）を格納する変数
tfidf_lst = [] # 特徴度を格納数する変数
cosSim_lst = [] # Cos類似度を格納する変数
rep_lst = [] # 単語のTF-IDF値を文でまとめた値を格納

# 1文に対するTFを求める。
def appearance(tagger): # リストの文字列
    # 出現数をカウントする
    text = tagger.copy()
    appearances_list = []
    length = len(text)
    for i in range(length):
        num = 1 # 単語の出現数
        j = 0
        for j in range(length):
            if i == j: continue # 同じ場所はカウントしない
            if text[i] == text[j]:
                #text.pop(j) # 比較済みのものは削除
                num+=1 # カウント
        appearances_list.append(num)
    return appearances_list

def wordcnt(document):
    sum = 0
    for s in document: sum += len(s)
    return sum

def cosSim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def median(i):
    i = i.copy()
    list.sort(i)
    return i[int(len(i)/2)-1]

# 文書ファイル読み込み
with open("TEXT.txt", "r", encoding="utf-8") as f:
    text = f.read()

####### テキスト整形 #######
text = text.replace("\n", "").replace("―", "")
text = text.split("。")
text.pop() # 最後の空の配列を削除
sentence_length = len(text)
print("Sentence：", str(sentence_length))


####### 単語で分割 #######
for sentence in text:
    tagger = MeCab.Tagger("-Owakati")
    tagger = tagger.parse(sentence)
    tagger = tagger.split(" ")
    tagger.remove("\n")
    document.append(tagger)

print("word：", wordcnt(document))


####### TF（頻出度）を求める #######
for sentence in document:
    freq = appearance(sentence) # 単語の出現数
    tf_lst.append([i/len(freq) for i in freq]) # 頻出度を格納する


####### IDF（希少度）を求める #######
for i in range(sentence_length):
    sentence_idf = []
    for word in document[i]: # document > sentence > word
        cnt = 1
        for j in range(sentence_length):
            if i == j: continue
            if word in document[j]: cnt += 1
        sentence_idf.append( math.log(sentence_length / cnt) ) # log(全行数 / その単語が出現する行数)
    idf_lst.append(sentence_idf)

print("TF list Size：", len(tf_lst))
print("IDF list Size：", len(idf_lst))


####### TF・IDFで特徴度を求める #######
for i in range(len(tf_lst)):
    tfidf = []
    for j in range(len(tf_lst[i])):
        tfidf.append(tf_lst[i][j] * idf_lst[i][j])
    tfidf_lst.append(tfidf)

print("TF・IDF list Size：", len(tfidf_lst))


####### TF・IDFの最大値を取り出す #######
for _tfidf in tfidf_lst:
    rep_sum = 0
    for rep in _tfidf:
        rep_sum += rep # 単語のTF-IDF値を足し合わせる
    rep_lst.append(rep_sum) 

print("Max of TF・IDF：", max(rep_lst))
print("Min of TF・IDF：", min(rep_lst))


####### Cos類似度を求める #######
res = 0
for i in range(len(tfidf_lst)):
    cosSim_words = []
    for j in range(len(tfidf_lst)):
        # 0を加えてベクトルの数を合わせる
        if len(tfidf_lst[i]) < len(tfidf_lst[j]):
            for k in range(len(tfidf_lst[j])-len(tfidf_lst[i])): tfidf_lst[i].append(0)
        elif len(tfidf_lst[i]) > len(tfidf_lst[j]):
            for k in range(len(tfidf_lst[i])-len(tfidf_lst[j])): tfidf_lst[j].append(0)
        # Cos類似度計算
        res = cosSim(tfidf_lst[i], tfidf_lst[j])
        cosSim_words.append(res)
        # ゼロベクトルの削除
        while 0 in tfidf_lst[i]: tfidf_lst[i].remove(0)
        while 0 in tfidf_lst[j]: tfidf_lst[j].remove(0)
    # 単語ベクトルのCos類似度の平均
    sum = 0
    for cw in cosSim_words: sum += cw
    cosSim_lst.append(cw/len(cosSim_words))
print("Cos Sim list Size：",len(cosSim_lst))


####### 要約をする（特徴度最大） #######
# 1行目：特徴度が最大な文をつ選ぶ
for i in range(len(rep_lst)):
    if max(rep_lst) == rep_lst[i]:
        print("TF-IDF：{:.5f}, Cos Sim：{:.5f} | {}".format(rep_lst[i], cosSim_lst[i], text[i]))
        break

# 2行目：特徴度が大きく、Cos類似度が低い文を選ぶ
# 代表値と(1 - Cos類似度)を掛け合わせることによって特徴を維持しつつベクトルを1つにすることができる。最大値を選ぶ。
# 1からCos類似度を引くことによって代表値に重みを持たせる。
ap = [rep_lst[i] * (1-cosSim_lst[i]) for i in range(len(rep_lst))]
for i in range(len(rep_lst)):
    if min(ap) == rep_lst[i] * (1-cosSim_lst[i]):
        print("TF-IDF：{:.5f}, Cos Sim：{:.5f} | {}".format(rep_lst[i], cosSim_lst[i], text[i]))
        break

# 3行目：特徴度が大きく、Cos類似度が低い文を選ぶ
cosSim_lst.pop(i) # 最も低いCos類似度を削除
rep_lst.pop(i)    # 同じ場所を削除して合わせる
text.pop(i)       # 同じ場所を削除して合わせる
ap = [rep_lst[i] * (1-cosSim_lst[i]) for i in range(len(rep_lst))]
for i in range(len(rep_lst)):
    if min(ap) == rep_lst[i] * (1-cosSim_lst[i]):
        print("TF-IDF：{:.5f}, Cos Sim：{:.5f} | {}".format(rep_lst[i], cosSim_lst[i], text[i]))
        break