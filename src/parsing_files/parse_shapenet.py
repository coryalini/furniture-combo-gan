import os
import json
import zipfile

def get_all_items(dir_num, base_path):
    file_names = []

    rel_path = base_path + str(dir_num)
    # print(list(filter(os.path.isdir, os.listdir(rel_path))))

    for root, dirs, files in os.walk(rel_path):
        if not("images" in root or "models" in root):
            file_names.append(root)

    return file_names


def zipdir(path, ziph):
    print("in here")
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def zipit(dir_list, zip_name):
    print("in there")
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    zipf.close()

if __name__ == '__main__':

    base_path = "ShapeNetCore.v2/"
    file = open(base_path+"taxonomy.json")
    taxonomy = json.load(file)

    chairs_dir_names = []
    tables_dir_names = []
    couch_dir_names = []

    for t in taxonomy:
        if t["name"] == "chair":# or t["name"] == "armchair":
            dir_num = t["synsetId"]
            chairs_dir_names = get_all_items(dir_num, base_path)
            zipit(chairs_dir_names, base_path+"chair_all.zip")
            print("zipped chairs")
        if t["name"] == "table":
            dir_num = t["synsetId"]
            tables_dir_names = get_all_items(dir_num, base_path)
            zipit(tables_dir_names, base_path+"table_all.zip")
            print("zipped table")
        if t["name"] == "sofa,couch,lounge":# or t["name"] == "convertible,sofa bed":
            dir_num = t["synsetId"]
            couch_dir_names = get_all_items(dir_num, base_path)
            zipit(couch_dir_names, base_path+"couch_all.zip")
            print("zipped couch")

    # "synsetId": "03001627",
    # "name": "chair",

    # "synsetId": "04379243",
    # "name": "table",

    # "synsetId": "04256520",
    # "name": "sofa,couch,lounge",