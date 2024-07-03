import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_dir, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt'), 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')

# Define the classes
classes = ["red light", "yellow light", "green light"]

# Paths
current_path = os.getcwd()
xml_dir = os.path.join(current_path,'dataset/annotations/xml')
output_dir = os.path.join(current_path, 'dataset/annotations/output')
os.makedirs(output_dir, exist_ok=True)

# Convert all XML files
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        convert_annotation(os.path.join(xml_dir, xml_file), output_dir, classes)