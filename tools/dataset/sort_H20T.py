import os
import shutil

os.chdir(os.path.dirname(os.path.realpath(__file__)))
if not os.path.exists("W"):
    os.makedirs("W")
if not os.path.exists("T"):
    os.makedirs("T")
if not os.path.exists("Z"):
    os.makedirs("Z")

for name in os.listdir("."):
    if name.endswith("T.JPG"):
        shutil.move(name,os.path.join("T",name))
    elif name.endswith("W.JPG"):
        shutil.move(name,os.path.join("W",name))
    elif name.endswith("Z.JPG"):
        shutil.move(name,os.path.join("Z",name))