"""
trafficlight.py
---------------
Convierte anotaciones de objetos desde formato XML (Pascal VOC) al formato
de texto utilizado por YOLO (bounding boxes normalizadas).

Clases soportadas:
    0 - red light    (luz roja)
    1 - yellow light (luz amarilla)
    2 - green light  (luz verde)

Estructura de directorios esperada:
    dataset/
    ├── annotations/
    │   ├── xml/        ← archivos .xml de entrada
    │   └── output/     ← archivos .txt generados (creado automáticamente)
    └── images/

Uso:
    python trafficlight.py
"""

import os
import xml.etree.ElementTree as ET


# Clases del modelo — el índice corresponde al class_id en YOLO
CLASSES = ["red light", "yellow light", "green light"]


def convert(size, box):
    """
    Convierte coordenadas absolutas de bounding box al formato normalizado de YOLO.

    Args:
        size (tuple): (ancho, alto) de la imagen en píxeles.
        box  (tuple): (xmin, xmax, ymin, ymax) en píxeles.

    Returns:
        tuple: (x_center, y_center, width, height) normalizados a [0, 1].
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def convert_annotation(xml_file, output_dir, classes):
    """
    Parsea un archivo XML de anotación y escribe el .txt equivalente para YOLO.

    Args:
        xml_file   (str):  Ruta al archivo .xml de entrada.
        output_dir (str):  Directorio donde se guardará el .txt resultante.
        classes    (list): Lista de nombres de clases (define el mapeo a class_id).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size_node = root.find("size")
    img_w = int(size_node.find("width").text)
    img_h = int(size_node.find("height").text)

    out_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(xml_file))[0] + ".txt",
    )

    with open(out_path, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((img_w, img_h), b)
            out_file.write(f"{cls_id} " + " ".join(str(v) for v in bb) + "\n")


if __name__ == "__main__":
    current_path = os.getcwd()
    xml_dir = os.path.join(current_path, "dataset", "annotations", "xml")
    output_dir = os.path.join(current_path, "dataset", "annotations", "output")
    os.makedirs(output_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    print(f"Convirtiendo {len(xml_files)} archivos XML...")

    for xml_file in xml_files:
        convert_annotation(os.path.join(xml_dir, xml_file), output_dir, CLASSES)

    print(f"Conversión completada. Archivos guardados en: {output_dir}")
