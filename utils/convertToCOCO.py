import os, json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

CLASS_NAMES = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']
CLASS2ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}  # COCO category_id starts at 1

def convert_voc_to_coco(image_ids, image_dir, anno_dir):
    images, annotations = [], []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
    ann_id = 1

    for img_id, file_id in enumerate(tqdm(image_ids)):
        img_file = f"{file_id}.jpg"
        xml_file = os.path.join(anno_dir, f"{file_id}.xml")
        img_path = os.path.join(image_dir, img_file)

        width, height = Image.open(img_path).size

        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        root = ET.parse(xml_file).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in CLASS2ID:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS2ID[name],
                "bbox": [xmin, ymin, w, h],  # ✅ COCO format
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def run_conversion(dataset_root):
    for split in ['train', 'test']:
        list_path = os.path.join(dataset_root, 'ImageSets', 'Main', f'{split}.txt')
        with open(list_path) as f:
            image_ids = [line.strip() for line in f.readlines()]

        coco = convert_voc_to_coco(
            image_ids,
            os.path.join(dataset_root, 'JPEGImages'),
            os.path.join(dataset_root, 'Annotations')
        )

        os.makedirs(os.path.join(dataset_root, 'annotations'), exist_ok=True)
        output_path = os.path.join(dataset_root, 'annotations', f'watercolor_{split}.json')
        with open(output_path, 'w') as f:
            json.dump(coco, f, indent=2)

if __name__ == "__main__":
    names = ['clipart', 'watercolor', 'comic']

    for name in names:
        run_conversion(f'./data/{name}')
