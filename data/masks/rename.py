import os

files = os.listdir('.')
for file_name in files:
    portion = os.path.splitext(file_name)
    if portion[1] == ".gif":
        new_name = portion[0]-"_masks" + ".jpg"

        os.rename(file_name, new_name)
