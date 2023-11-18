import sys
import csv

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"
csv_file2 = csv_file_path + "token.csv"
csv_file3 = csv_file_path + "token2vec.csv"

csv_file_header = ['ob_models_input', 'origin_models_label']

with open(csv_file1, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_header)
csv_file.close()
with open(csv_file2, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_header)
csv_file.close()
with open(csv_file3, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_header)
csv_file.close()