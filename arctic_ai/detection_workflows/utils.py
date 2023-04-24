"""Nuclei Detection Utils
==========
Additional functions to support cell detection. 
"""
import os, glob
import cv2
import numpy as np
import pickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from detectron2.data import transforms as T
import pycocotools
from detectron2.structures import BoxMode
from fvcore.transforms.transform import Transform, TransformList
from PIL import Image, ImageDraw, ImageEnhance
from skimage.color import rgb2hed, hed2rgb

def draw_point(image, x, y, size):
    draw = ImageDraw.Draw(image)
    draw.ellipse((x - size//2, y - size//2, x + size//2, y + size//2), fill = 'yellow', outline ='yellow')
    return image

def com(bmask):
    m = bmask / np.sum(np.sum(bmask))

    dx = np.sum(m, 0) # there is a 0 here instead of the 1
    dy = np.sum(m, 1)

    # expected values
    xmean = np.sum(dx * np.arange(m.shape[1]))
    ymean = np.sum(dy * np.arange(m.shape[0]))
    return xmean, ymean

def get_identifier(path):
    return os.path.splitext(os.path.basename(path))[0].replace("_ASAP","").replace("_fixed", "").replace("_spline_data", "").replace("_id_mask", "").replace("_updated", "")

def get_class_map():
    return {
        "BCC": 0, 
        "bcc" : 0, 
        "bcc nuclei" : 0, 
        "BCC Nuclei": 0, 
        "Fibroblast": 1, 
        "fibroblast" : 1, 
        "Fibroblast nuclei" : 1,
        "Fibroblast Nuclei" : 1, 
        "fibroblast nuclei" : 1, 
        "Hair Follicle": 2, 
        "hair follicle" : 2,
        "hair follicle nuclei" : 2, 
        "Hair Follicle Nuclei" : 2,
        "Hair Follicle" : 2,
        "Epidermal Keratinocyte": 3, 
        "epidermal keratinocyte": 3, 
        "Inflammatory Nuclei": 4,
        "Inflammation": 4,
        "Inflammatory": 4,
        "inflammatory": 4
    }

def get_groups_dict():
    return {
        "all": {
            "map": {
                0: "BCC",
                1: "Fibroblast",
                2: "Hair Follicle",
                3: "Epidermal Keratinocyte",
                4: "Inflammatory",
            },
            "colors": ["#ff0000", "#ffff7f", "#aa00ff", "#55aaff", "#64FE2E"],
            "groups": [
                "BCC",
                "Fibroblast",
                "Epidermal Keratinocyte",
                "Hair Follicle",
                "Inflammatory",
            ],
        },
        "cnn" : {
            "map": {
                0: "BCC",
                1: "Fibroblast",
                2: "Hair Follicle",
                3: "Epidermal Keratinocyte",
            },
            "colors": ["#ff0000", "#ffff7f", "#aa00ff", "#55aaff"],
            "groups": [
                "BCC",
                "Fibroblast",
                "Epidermal Keratinocyte",
                "Hair Follicle",
            ],
        },
        "two": {
            "map": {0: "Malignant", 1: "Benign"},
            "colors": ["#ff0000", "#ffff7f"],
            "groups": ["Benign", "Malignant"],
        },
        "single": {"map": {0: "Nuclei"}, "colors": ["#ff0000"], "groups": ["Nuclei"]},
    }

def load_patches(path, file_name, patch_folder_name="patches", mask_folder_name="masks"):
    patches = np.load(os.path.join(path, f"{patch_folder_name}/{file_name}")) # change to suit your needs
    masks = np.load(os.path.join(path, f"{mask_folder_name}/{file_name}")) # chnage to suit your needs
    return patches, masks

def get_types_dict():
    if not Path("id_to_type.pkl").exists():
        raise NotImplementedError# id_to_types()
    return pickle.load(open("id_to_type.pkl", "rb"))

def rgb_encode(patch_id):
    B = patch_id // 256 ** 2
    patch_id -= B * 256 ** 2
    G = patch_id // 256
    patch_id -= G * 256
    R = patch_id
    return (R, G, B)

def add_segmentation_pic(seg_image, segmentation, segmentation_color):
    seg_image[segmentation] = segmentation_color

def generate_images(patch_list, coco_save_directory, preprocess=None):
    print("Generating images...")
    for patch_index, patch in enumerate(tqdm(patch_list)):
        patch_index += 1 # change 0-indexed outputs to 1-indexed
        patch_id = (patch_index) * 1000 + 131
        save_file_name = os.path.join(coco_save_directory, str(patch_id) + ".jpg") # use jpg because detectron considers this normal (pngs for annotations)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        if preprocess:
            patch = preprocess(patch)
        cv2.imwrite(save_file_name, patch)
    
def get_draw_groups():
    return {'131_B1e' : ["BCC", "Hair Follicle", "Fibroblast", "Epidermal Keratinocyte"],'51_D1b' : ["BCC Nuclei", "Hair Follicle Nuclei", "Fibroblast nuclei"],'41_A2c' : ["BCC Nuclei", "Hair Follicle Nuclei", "Fibroblast nuclei", "Inflammatory Nuclei"]}

def fix_xml_ids(xmlname, save_path):
    tree = ET.parse(xmlname)
    root = tree.getroot()

    for i, annotation in enumerate(root[0]):
        annotation.attrib["Name"] =  f"Annotation {i}"

    new_xml = ET.tostring(root)

    with open(save_path, "wb") as f:
        f.write(new_xml)

def visualize_augment(image_transformed, boxes_transformed, sem_seg_transformed):
    # Visualize the augmented images:
    for index in range(len(boxes_transformed)):
        x1, y1, x2, y2 = [int(i) for i in boxes_transformed[index]]
        color = list(np.random.random(size=3) * 256)
        cv2.rectangle(image_transformed, (x1, y1), (x2, y2), color, 1)
        write_segmentation(image_transformed, sem_seg_transformed[index], color)

def write_segmentation(image, seg, color=(255, 0, 0)):
    width, height = seg.shape
    for i in range(width):
        for k in range(height):
            if seg[i][k] == 1:
                image[i][k] = color

def augment_annotation_detectron(augmentation, annotation, augment_index, augment_index_max):
    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            sem_seg = []
            for segmentation in self.sem_seg:
                sem_seg.append(tfm.apply_segmentation(segmentation))
            self.sem_seg = sem_seg

    T.AugInput.transform = transform

    image_id = annotation['image_id']
    file_name = annotation['file_name']
    iscrowd = annotation['iscrowd']
    annotations = annotation['annotations']

    if len(annotations) == 0:
        return []

    box_mode = annotations[0]['bbox_mode']
    boxes = []
    sem_seg = []
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for c, annotation in enumerate(annotations):
        category_id = annotation['category_id']
        box = annotation['bbox']
        
        if box_mode == BoxMode.XYWH_ABS:
            xmin, ymin, width, height = box
            box = (xmin, ymin, xmin + width, ymin + height) # Convert to XYXY

        boxes.append(box)

        segmentation = annotation['segmentation']
        segmentation = pycocotools.mask.decode(segmentation)
        sem_seg.append(segmentation)
        
    boxes = np.array(boxes) # must be in XYXY for augmentation

    # Define the augmentation input ("image" required, others optional):
    input = T.AugInput(image, boxes=boxes, sem_seg=sem_seg)

    # Apply the augmentation:
    transform = augmentation(input)  # type: T.Transform
    image_transformed = input.image  # new image
    boxes_transformed = input.boxes  # new boxes
    sem_seg_transformed = input.sem_seg  # new semantic segmentation
    new_image_id = image_id + augment_index_max * 10 + augment_index + 1
    file_name = os.path.join(os.path.split(file_name)[0], '%s.jpg' % new_image_id)

    augmented_data = {
        "image_id": new_image_id,
        "width": image_transformed.shape[1], # switch width and height
        "height": image_transformed.shape[0],
        "file_name": file_name,
        "iscrowd" : iscrowd,
        "annotations": []
    }
    augmented_annotation = []

    # draw box augmentation and segmentations
    for index, annotation in enumerate(annotations):
        category_id = annotation['category_id']
        bbox_mode = annotation['bbox_mode']
        bounding_box = [float(i) for i in boxes_transformed[index]]

        if box_mode == BoxMode.XYWH_ABS:
            xmin, ymin, xmax, ymax = box
            bounding_box = (xmin, ymin, xmax - xmin, ymax - ymin) # XYWH

        current_annotation = {
            "category_id": category_id,
            "segmentation": pycocotools.mask.encode(np.asarray(sem_seg_transformed[index], order="F")),
            "bbox": bounding_box,
            "bbox_mode": box_mode
        }
        current_annotation["segmentation"]["counts"] = current_annotation["segmentation"]["counts"].decode("utf-8")
        augmented_annotation.append(current_annotation)

    # visualize_augment(image_transformed, boxes_transformed, sem_seg_transformed)

    augmented_data['annotations'] = augmented_annotation
    cv2.imwrite(file_name, image_transformed)
    
    return augmented_data
    
def augment_annotation_coco(augmentation, augment_index, annotation, current_image=None):
    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            sem_seg = []
            for segmentation in self.sem_seg:
                sem_seg.append(tfm.apply_segmentation(segmentation))
            self.sem_seg = sem_seg

    T.AugInput.transform = transform

    image_id = annotation['image_id']
    file_name = annotation['file_name']
    iscrowd = annotation['iscrowd']
    annotations = annotation['annotations']

    if len(annotations) == 0:
        return []

    box_mode = annotations[0]['bbox_mode']
    boxes = []
    sem_seg = []
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for c, annotation in enumerate(annotations):
        category_id = annotation['category_id']
        box = annotation['bbox']
        
        if box_mode == BoxMode.XYWH_ABS:
            xmin, ymin, width, height = box
            box = (xmin, ymin, xmin + width, ymin + height) # Convert to XYXY

        boxes.append(box)

        segmentation = annotation['segmentation']
        segmentation = pycocotools.mask.decode(segmentation)
        sem_seg.append(segmentation)
        
    boxes = np.array(boxes) # must be in XYXY for augmentation

    # Define the augmentation input ("image" required, others optional):
    input = T.AugInput(image, boxes=boxes, sem_seg=sem_seg)

    # Apply the augmentation:
    transform = augmentation(input)  # type: T.Transform
    image_transformed = input.image  # new image
    boxes_transformed = input.boxes  # new boxes
    sem_seg_transformed = input.sem_seg  # new semantic segmentation
    new_image_id = image_id + augment_index + 1
    file_name = os.path.join(os.path.split(file_name)[0], '%s.jpg' % new_image_id)

    augmented_data = {
        "image_id": new_image_id,
        "width": image_transformed.shape[1], # switch width and height
        "height": image_transformed.shape[0],
        "file_name": file_name,
        "iscrowd" : iscrowd,
        "annotations": []
    }
    augmented_annotation = []

    # draw box augmentation and segmentations
    for index, annotation in enumerate(annotations):
        category_id = annotation['category_id']
        bbox_mode = annotation['bbox_mode']
        bounding_box = [float(i) for i in boxes_transformed[index]]

        if box_mode == BoxMode.XYWH_ABS:
            xmin, ymin, xmax, ymax = box
            bounding_box = (xmin, ymin, xmax - xmin, ymax - ymin) # XYWH

        area = bin_mask.sum()
        
        current_annotation = {
            "category_id": category_id,
            "segmentation": pycocotools.mask.encode(np.asarray(sem_seg_transformed[index], order="F")),
            "bbox": bounding_box,
            "bbox_mode": box_mode
        }
        current_annotation["segmentation"]["counts"] = current_annotation["segmentation"]["counts"].decode("utf-8")
        augmented_annotation.append(current_annotation)

    # visualize_augment(image_transformed, boxes_transformed, sem_seg_transformed)

    augmented_data['annotations'] = augmented_annotation
    cv2.imwrite(file_name, image_transformed)
    
    return augmented_data

def fix_model_xml(xmlname, save_path, L):
    tree = ET.parse(xmlname)
    root = tree.getroot()

    ann = root.find('Annotations')
    annotations = list(enumerate(ann))
    
    for i, annotation in annotations:
        if i >= L:
            ann.remove(annotation)
        else:
            annotation.attrib["Color"] = "#F4FA58"
            
    colors = ["#ff0000", "#ffff7f", "#aa00ff", "#55aaff", "#64FE2E"]
    groups = ["BCC", "Sebaceous Gland", "Fibroblast", "Epidermal Keratinocyte", "Hair Follicle"]
    
    anngroups = ET.SubElement(root, "AnnotationGroups")
    
    for name, color in zip(groups, colors):
        group = ET.SubElement(anngroups, "Group")
        group.attrib["Color"] = color
        group.attrib["Name"] = name
        group.attrib["PartOfGroup"] = "None"
        attr = ET.SubElement(group, "Attributes")
   
    new_xml = ET.tostring(root)

    with open(save_path, "wb") as f:
        f.write(new_xml)
        
def shorten_xml(xmlname, save_path, L):
    tree = ET.parse(xmlname)
    root = tree.getroot()

    ann = root.find('Annotations')
    annotations = list(enumerate(ann))
    
    for i, annotation in annotations:
        if i >= L:
            ann.remove(annotation)
            
    new_xml = ET.tostring(root)

    with open(save_path, "wb") as f:
        f.write(new_xml)
        
def change_class(xmlname, save_path, orig, new):
    # replaces class orig with class new 
    tree = ET.parse(xmlname)
    root = tree.getroot()
    
    for i, annotation in enumerate(root[0]):
        cls = annotation.attrib['PartOfGroup']
        if (cls == orig):
            cls = new
        annotation.attrib['PartOfGroup'] = cls
            
    new_xml = ET.tostring(root)

    with open(save_path, "wb") as f:
        f.write(new_xml)
