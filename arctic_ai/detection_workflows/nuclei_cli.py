'''
Nuclei Detection  
==========

Example command:

python cli.py detect_from_patches \
--predictor_dir=. \
--predictor_file=./ArcticAI_Detection/output_a=0_128_more_gr_pretrained_d2s/model_final.pth \
--patch_file=./ArcticAI_Detection/ex4/spline/patches/131_B1e.npy \
--threshold=0.05

'''
from itertools import product
from xml.dom import minidom
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import pickle
import os
import fire

# local
from .predict_detection import load_predictor, PanopticNucleiPredictor
from .utils import com

#1. Iterate through patches 
#2. Run model on patches
#3. Save Results in XML

#cv2.findContours gives lists of [col,row] points

def extract_nuclei_patch(image, mask, coord, size):
    '''
    Get a subsection of the image with nuclei at the center
    Args:
        image: the whole slide image
        mask: nuclei mask 
        coord: position if the form (row, column) of the top left corner of the mask in the image
        size: size of the patch to extract (size x size)
    '''
    xmean,ymean = com(mask) # column, row
    xmean = int(xmean)
    ymean = int(ymean)
    
    box = [coord[0] + ymean - size/2, coord[1] + xmean - size/2, coord[0] + ymean + size/2, coord[1] + xmean + size/2]
    box = [int(x) for x in box]
    patch = image[box[0]:box[2], box[1]:box[3], :]
    return patch

def run(predictor, patches, threshold=0.2):
    '''
    Returns a list of tuples. Tuples are of the form (masks for patch, associated labels for patch)
    Args:
        predictor: model
        patches: must be BGR
        duplicate: if set to False, removes duplication
    '''
    mp = predictor.classes['map']
    out = []
    for patch in tqdm(patches):
        masks, _, labels = predictor.predict(patch, threshold=threshold)
        assert(len(masks) == len(labels))
        labels = [mp[l] for l in labels]
        out.append((masks, labels))
    return out

def run_for_patch_with_cnn(predictor, cnn, image, coord, size, duplicate=False): # coord and size define the patch
    '''
    Args:
        predictor: model
        cnn: class predictor
        image: whole slide image
        coord: (row, col) of the top left corner of the patch 
        size: size of patches to pass through predictor
        duplicate: if set to False, removes duplication
        
    '''
    row,col = coord
    patch = image[row:row+size,col:col+size]
    patch = np.flip(patch, 2) # makes it BGR
    mp = cnn.classes['map']
    masks, _, _labels = predictor.predict(patch)
    good_masks = []
    labels = []
    if duplicate is False:
        masks, _labels = remove_duplicates(masks, _labels)
    for mask in masks:
        try: 
            int_lbl = cnn.predict(extract_nuclei_patch(image, mask, coord, cnn.size))
            good_masks.append(mask)
            labels.append(mp[int_lbl])
        except:
            print("bad output")
    return good_masks, labels
        
def run_with_cnn(predictor, cnn, image, coords, size):
    print("here")
    out = []
    for coord in tqdm(coords):
        out.append(run_for_patch_with_cnn(predictor, cnn, image, coord, size))
    return out
        
def export_npy(out, savenpy='pred.npy'):
    num = 1
    all_masks = []
    detection_info = {}
    for masks, labels in tqdm(out):
        if (len(masks) > 0):
            all_mask = np.zeros(masks[0].shape)
            for mask, label in zip(masks, labels):
                all_mask[mask==1] = num
                detection_info[num] = label
                num+=1
            all_masks.append(all_mask)
    np.save(savenpy, np.stack(all_masks))
    pickle.dump(detection_info, open(os.path.splitext(savenpy)[0]+'.pkl', 'wb'))
    # save the all

def export_xml(coords, out, groups, colors, savexml='out.xml'):
    c=0
    top = ET.Element('ASAP_Annotations')
    annotations = ET.SubElement(top, "Annotations")
    for coord, (masks, labels) in zip(coords, out):
        row, col = coord
        for mask, label in zip(masks, labels):
            annotation = ET.SubElement(annotations, "Annotation")
            annotation.attrib["Name"] = f"Annotation {c}"; c+=1
            annotation.attrib["Type"] = "Polygon"
            annotation.attrib["PartOfGroup"] = label
            annotation.attrib["Color"] = "#F4FA58"

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            try: 
                assert(len(contours) > 0)

                ind = 0

                for i in range(len(contours)):
                    if(contours[i].shape[0] > contours[ind].shape[0]): 
                        ind = i
                contour = contours[ind]

                assert(contour.shape[0] >= 3)

                contour = contour.squeeze()

                coords = ET.SubElement(annotation, "Coordinates")
                x = 0
                for i, point in enumerate(contour):
                    coord = ET.SubElement(coords, "Coordinate")
                    coord.attrib["Order"] = str(i)
                    coord.attrib["X"] = str(col + point[0])
                    coord.attrib["Y"] = str(row + point[1])
                    x+=1
                # Next loop is only for the last point
                for i, point in enumerate(contour):
                    coord = ET.SubElement(coords, "Coordinate")
                    coord.attrib["Order"] = str(x)
                    coord.attrib["X"] = str(col + point[0])
                    coord.attrib["Y"] = str(row + point[1])
                    break
            except:
                annotations.remove(annotation)
                print("Bad output:", contours)

    anngroups = ET.SubElement(top, "AnnotationGroups")

    for name, color in zip(groups, colors):
        group = ET.SubElement(anngroups, "Group")
        group.attrib["Color"] = color
        group.attrib["Name"] = name
        group.attrib["PartOfGroup"] = "None"
        attr = ET.SubElement(group, "Attributes")

    save_xml(top, savexml)

def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def save_xml(top, filename):
    xml = prettify(top)
    
    with open(filename, "w") as f:
        f.write(xml)

def detect_from_patches(predictor_dir="./", predictor_file="", patch_file="", classifier_type=PanopticNucleiPredictor, threshold=0.05, savenpy='pred.npy', savexml=None, patch_coords=None): 
    '''
    Args:
        predictor_dir (str): path to model folder
        predictor_file (str): filename of model in the folder (don't include path to folder)
        patch_file (str): path to an npy stack of patches
        classifier_type (BasePredictor): class (from predict.py)
        panoptic (bool): whether the model performs panoptic segmentation. If false, it is assumed to do instance segmentation.
        n (int): number of classes the model classifies into 
        threshold (float): threshold to use for model 
        savenpy (str): Path to file to which to save npy output. The output is a numpy stack of the the masks for each patch. In the mask, nuclei are given a non-zero integer, the ID of the the instance it is a part of. A pickled dictionary mapping the instance ID to the class label is also outputted. If savenpy=None, predictions are not saved in an npy format. 
        savexml (str): Path to file to which to save xml output (ASAP format). If savexml=None, predictions are not saved in an xml format. 
        patch_coords (str): an extra pkl file which specifies the x,y (x is row, y is col) metadata for the patches. Must be provided if exporting to xml, since location is a part of the ASAP format
    '''

    print("Loading files...")

    predictor = load_predictor(classifier_type, predictor_dir, predictor_file, threshold=threshold)
    patches = np.load(patch_file)
    patches = np.flip(patches, 3)
    print("Running model...")
    
    out = run(predictor, patches)
    
    if savenpy:
        print("Saving predictions to an npy stack and label dictionary...")
        export_npy(out, savenpy=savenpy)
    if savexml:
        print("Saving predictions to an ASAP xml...")
        assert(patch_coords is not None) # must give coordinates to 
        coords = pickle.load(open(patch_coords, 'rb'))[['x', 'y']].values
        export_xml(coords, out, predictor.classes['groups'], predictor.classes['colors'], savexml=savexml)

def extract(image, coords, size):
    patches = np.stack([image[x:x+size,y:y+size] for x,y in tqdm(coords)]) # x is really row, y is really col
    patches = np.flip(patches, 3)
    return patches

def detect_from_wsi(predictor, patches, coords):
    raise NotImplementedError

def get_expanded_dataset_slide(slide):
    from pathpretrain.utils import load_image
    return load_image(os.path.join(
            "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/arctic_nuclei/arctic_WSI_dataset_expanded/inputs/",
            slide+".npy"))

def get_diverse_patches(slide, size=128, thresh=0.5):
    '''
    get benign, inflammatory, and bcc patch for a slide
    '''
    metadata = pickle.load(open('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/arctic_nuclei/arctic_WSI_dataset_expanded/tumor_annot_gnn_annot.pkl', 'rb'))
    df = metadata.loc[metadata['ID'] == slide]
    benign = df.loc[(df['annotation'] == 'benign') & (df['inflammation'] <thresh)]
    bcc = df.loc[(df['annotation'] == 'bcc') & (df['inflammation'] < thresh)]
    inflammatory = df.loc[df['inflammation'] >thresh]
    coords = []
    if len(benign) == 0:
        print("No benign patches on this slide. Skipping...")
    else:
        coord = benign.sample()[['x', 'y']].values[0]
        coords.extend(list(product(range(coord[0], coord[0] + 256, size), range(coord[1], coord[1] + 256, size)))) # col
    if len(bcc) == 0:
        print("No bcc patches on this slide. Skipping...")
    else:
        coord = bcc.sample()[['x', 'y']].values[0]
        coords.extend(list(product(range(coord[0], coord[0] + 256, size), range(coord[1], coord[1] + 256, size)))) # col
    if len(inflammatory) == 0:
        print("No inflammatory patches on this slide. Skipping...")
    else:
        coord = inflammatory.sample()[['x', 'y']].values[0]
        coords.extend(list(product(range(coord[0], coord[0] + 256, size), range(coord[1], coord[1] + 256, size)))) # col
    coords = np.stack(coords)
    return coords

def make_expanded_dataset_xml(predictor, date, slide, coords, size=128, cnn=None):
    print("Loading image...")
    image=get_expanded_dataset_slide(slide)
    print("Running model...")
    if cnn is None:
        patches = extract(image, coords, size)
        out = run(predictor, patches)
    else:
        out = run_with_cnn(predictor, cnn, image, coords, size)
    num = 0
    savexml = os.path.join("../model_xmls/", slide + "_" + date + '_thresh_' + str(predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)[2:])
    while (os.path.exists(f"{savexml}_{num}.xml")):
        num+=1
    savexml = f"{savexml}_{num}.xml"
    export_xml(coords, out, predictor.classes['groups'], predictor.classes['colors'], 
            savexml=savexml)
         
    
def make_xmls_files(predictor, date, files, size=128): # surgeon-defined files containing boxes of interest
    slides = []
    boxes = []
    for file in files:
        slide = '_'.join(os.path.basename(file).split('_')[:2])
        new_boxes = get_boxes(file)
        for new_box in new_boxes:
            boxes.append(new_box)
            slides.append(slide)
    make_xmls_box(predictor, date, slides, boxes, size=size)
    
def make_xmls_box(predictor, date, slide_list, box_list, size=128, cnn=None):   
    # uses a box to generate the xml (slide_list and box_list)
    # currently only for the expanded dataset (should change)
    for slide, box in zip(slide_list, box_list):
        print("On slide:", slide)
        print("On box:", box)
        coords = np.stack(list(product(range(box[0], box[2], size), range(box[1], box[3], size))))
        print(len(coords), " total patches will be processed")
        make_expanded_dataset_xml(predictor, date, slide, coords, size=size, cnn=cnn)

def get_boxes(xmlpath):
    tree = ET.parse(os.path.join(xmlpath))
    root = tree.getroot()
    boxes = []
    for index, annotation in enumerate(root[0]):
        if (annotation.attrib['Type'] ==  "Rectangle"):
            minx, miny, maxx, maxy = float('inf'), float('inf'), 0, 0
            for coordinate in annotation[0]:
                x, y = float(coordinate.attrib["X"]), float(coordinate.attrib["Y"])
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
            box = [int(miny), int(minx), int(maxy), int(maxx)]
            boxes.append(box)
    return boxes

if __name__ == "__main__":
    fire.Fire({
        'detect_from_patches': detect_from_patches
    })
   