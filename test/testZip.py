docs = ["文档A", "文档B", "文档C"]
scores = [0.6, 0.8, 0.4]

z = zip(docs, scores)
for doc, score in z:
    print(doc, score)
