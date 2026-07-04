import dill as pickle

file = '../data/mimic3/output/voc_final.pkl'
with open(file, "rb") as f:
    idx2drug = pickle.load(f)

# 打印字典内容
#print(idx2drug)

diag_voc = idx2drug['med_voc']
idx2word = diag_voc.idx2word
#print(idx2word)
file2 = '../data/mimic3/input/idx2drug.pkl'
with open(file2, "rb") as f2:
    record = pickle.load(f2)

# 打印字典内容
#print(record)
unique_indices=[16, 40, 2, 37, 20, 13, 5, 3, 6, 4, 0, 50, 28, 21, 11, 88, 17, 35, 47, 10, 56, 29, 31, 18, 8, 107, 26]
matching_results = []
for idx in unique_indices:
    target_idx = idx  # 假设第三层的值是索引 k
    if target_idx in idx2word:
        code = idx2word[target_idx]
        matching_results.append({
            "索引 k": target_idx,
            "对应值 v": code
        })
    else:
        matching_results.append({
            "索引 k": target_idx,
            "对应值 v": None,
            "提示": "未在 diag_voc 中找到此索引"
        })

# 4. 输出结果
print("第三层序列1去重后的索引 k：", unique_indices)
print("\n索引 k 与 diag_voc 中值 v 的对应关系：")
for result in matching_results:
    if result["对应值 v"]:
        print(f"索引 {result['索引 k']} → 值：{result['对应值 v']}")
    else:
        print(f"索引 {result['索引 k']} {result['提示']}")
