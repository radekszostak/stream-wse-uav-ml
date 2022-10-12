import csv
import random
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

dir = "ort"
for name in os.listdir(dir):
    if "KOC1" in name:
        new_name = name.replace("KOC1", "GRO")
    if "KOC2" in name: 
        new_name = name.replace("KOC2", "RYB")
    if "AMO1" in name:    
        new_name = name.replace("AMO1", "AMO")
    os.rename(f"{dir}/{name}",f"{dir}/{new_name}")  

