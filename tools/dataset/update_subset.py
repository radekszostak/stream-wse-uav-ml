import csv
import random
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
new_rows = []
s_n = ""
n = 0
subsets = ["_","GRO21","RYB21","GRO20","RYB20","AMO18" ]
with open("dataset.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for i, row in enumerate(csv_reader):

        new_row = row
        new_row[0] = row[0]+".npy"
        new_rows.append(new_row)

with open("new_dataset.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    csv_writer.writerows(new_rows)
