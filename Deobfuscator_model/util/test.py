import csv

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "token_no_ob_info.csv"
csv_file2 = csv_file_path + "token_with_ob_info.csv"
csv_file3 = csv_file_path + "token_with_ob_info_new.csv"

csv_reader = csv.reader(open(csv_file1, encoding="utf-8"))
next(csv_reader)
no_ob_models = []
for row in csv_reader:
    s = row[0]
    s = (
        s.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .split(", ")
    )
    no_ob_model = []
    i = 0
    while i < len(s):
        tensor_info = []
        if s[i] == "torch.vtensor.literal":
            i += 1
            while s[i] != "EOOperation":
                tensor_info.append(s[i])
                i += 1
        if len(tensor_info) > 0:
            no_ob_model.append(tensor_info)
        i += 1
    no_ob_models.append(no_ob_model)

print("models count: ", len(no_ob_models))

csv_reader = csv.reader(open(csv_file2, encoding="utf-8"))
next(csv_reader)

model_sn = 0
for row in csv_reader:
    s = row[0]
    s = (
        s.replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .split(", ")
    )
    tensor_sn = 0
    i = 0
    while i < len(s):
        if s[i] == "torch.vtensor.literal":
            for mystr in no_ob_models[model_sn][tensor_sn]:
                s.insert(i + 1, mystr)
                i += 1
            tensor_sn += 1
        i += 1
    with open(csv_file3, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([s, row[1]])
    csv_file.close()
    model_sn += 1
