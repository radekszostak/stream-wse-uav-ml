import csv
import random
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
new_rows = []

with open("levels.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        random.seed(row[0])
        phase = "train" if random.random()>0.2 else "test"
        row.append(phase)
        new_rows.append(row)
    print(new_rows)
print(new_rows)
with open("new_levels.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    csv_writer.writerows(new_rows)
