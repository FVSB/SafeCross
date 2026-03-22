# Annotation Format

Standards for dataset annotation and conversion in SafeCross.

## Classes

Always use exactly these 3 classes in this order:

```python
CLASSES = ["red light", "yellow light", "green light"]
# class_id 0 = red light
# class_id 1 = yellow light
# class_id 2 = green light
```

Do not use 2-class subsets (red/green only) — this was a known inconsistency
in older code that has been fixed.

## Input Format: Pascal VOC XML

Source annotations are in Pascal VOC XML format:

```xml
<annotation>
  <size>
    <width>1280</width>
    <height>720</height>
  </size>
  <object>
    <name>red light</name>
    <bndbox>
      <xmin>540</xmin>
      <xmax>600</xmax>
      <ymin>100</ymin>
      <ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
```

## Output Format: YOLO TXT

Each `.txt` file has one line per detected object:
```
class_id x_center y_center width height
```
All values normalized to [0, 1].

## Conversion Formula

Use the following exact implementation (from `trafficlight.py`):

```python
def convert(size, box):
    """
    size: (image_width, image_height)
    box:  (xmin, xmax, ymin, ymax)  ← note: xmax before ymin
    returns: (x_center, y_center, width, height) normalized
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)
```

**Important:** The box tuple order is `(xmin, xmax, ymin, ymax)` — not the
typical XYXY order. This matches how the XML is read from `<bndbox>`.

## Directory Structure

```
dataset/
├── annotations/
│   ├── xml/        ← source XML files (input to trafficlight.py)
│   └── output/     ← generated YOLO .txt files
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/      ← YOLO .txt labels (moved from annotations/output/)
    ├── val/
    └── test/
```

Run conversion:
```bash
python trafficlight_detection/trafficlight.py
# Then manually move output/*.txt to labels/train/ (or val/test/)
```
