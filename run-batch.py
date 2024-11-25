# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


import sys
import numpy as np
# from pathlib import Path
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob

from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, User, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

import torch
import openvino as ov

import time
import cv2
import math
import csv

from io import BytesIO

from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil
from scipy.special import softmax 
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# TO RUN:
# python3.10 run-batch.py --cytomine_host "http://cytomine.imu.edu.my" --cytomine_public_key "71981a88-4e97-4551-a403-59d0dfccf5fd" --cytomine_private_key "36344b44-7539-4772-b1fb-d9ae7ec72182" --cytomine_id_project "77742" --cytomine_id_software "78621259" --cytomine_id_images "74197052" --cytomine_id_cell_term "1021760" --log_level "WARNING"

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "0.0.1"

def run(cyto_job, parameters):
    logging.info("----- Blood-class-DenseNet with OpenVino v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    confidence_threshold = 0.9
    NUM_THREADS = 10
    thread_local = threading.local()
    # modeltype = parameters.modeltype

    class_names = ['platelet-thrombocyte', 'rbc0-normal', 'rbc1-helmet-schistocyte', 'rbc10-stomatocyte', 'rbc11-target', 'rbc12-hypochromia', 'rbc13-macrocyte', 'rbc14-microcyte', 'rbc15-erythroblast', 'rbc2-ovalocyte', 'rbc3-spur-acanthocyte', 'rbc4-teardrop', 'rbc5-burr-echinocyte', 'rbc6-sickle', 'rbc8-pencil-elliptocyte', 'rbc9-spherocyte', 'wbc-ig0-immature-granulocyte', 'wbc-ig2-promyelocyte', 'wbc-ig3-myelocyte', 'wbc-ig4-metamyelocyte', 'wbc0-basophil', 'wbc0-eosinophil', 'wbc0-lymphocyte', 'wbc0-monocyte', 'wbc0-neutrophil']
    print(len(class_names))
    num_classes=len(class_names)   
    class_counters = {label: 0 for label in range(len(class_names))} 

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)
    # Extract term names and IDs into a dictionary
    term_mapping = {}

    for term in terms:
        # print(f"Original Term: Name={term.name}, ID={term.id}")
        term_mapping[term.name] = term.id
        
    # Function to normalize names
    def normalize_name(name):
        return name.lower().replace('-', '_').replace(' ', '')

    normalized_term_mapping = {normalize_name(name): term_id for name, term_id in term_mapping.items()}    

    # Verify normalized term mapping
    # print("\nNormalized Term Mapping:")
    # for name, term_id in normalized_term_mapping.items():
    #     print(f"Normalized Name={name}, ID={term_id}")

    # Create a mapping from label indices to term IDs with normalization
    label_to_term_id = {}
    for label, class_name in enumerate(class_names):
        normalized_class_name = normalize_name(class_name)
        term_id = normalized_term_mapping.get(normalized_class_name)
        
        if term_id is not None:
            label_to_term_id[label] = term_id
        else:
            print(f"Warning: Class '{class_name}' (normalized to '{normalized_class_name}') not found in ontology.")
            label_to_term_id[label] = None  # Handle as needed

    # Verify the final label-to-term mapping
    print("\nLabel to Term ID Mapping:")
    for label, term_id in label_to_term_id.items():
        class_name = class_names[label]
        if term_id:
            print(f"Label {label} ({class_name}): Term ID = {term_id}")
        else:
            print(f"Label {label} ({class_name}): Term ID = None (Mapping Issue)")

    start_time=time.time()

    # ----- load network ----
    def get_compiled_model_and_output_layer():
        if not hasattr(thread_local, "compiled_model"):
            # Load a new model instance for each thread
            core = ov.Core()
            ir_path = "/models/blood-class-v2_dn21adam_best_model_100ep.xml"
            model_ir = core.read_model(model=ir_path)
            thread_local.compiled_model = core.compile_model(model=model_ir, device_name='CPU')
            # Get the output layer of the compiled model
            thread_local.output_layer = thread_local.compiled_model.output(0)
        return thread_local.compiled_model, thread_local.output_layer
    # ------------------------

    print("Model successfully loaded!")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded!")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
    start_prediction_time=time.time()
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        def process_cell(roi, roi_id, roi_location, id_image, confidence_threshold, label_to_term_id, working_path, project, is_algo):
            try:
                imageinfo = ImageInstance(id=id_image, project=parameters.cytomine_id_project)
                imageinfo.fetch()
                wsi_width=imageinfo.width
                wsi_height=imageinfo.height
                roi_geometry = wkt.loads(roi_location)
                min_x=roi_geometry.bounds[0]
                min_y=roi_geometry.bounds[1]
                max_x=roi_geometry.bounds[2]
                max_y=roi_geometry.bounds[3]    
                roi_width=int(max_x - min_x)
                roi_height=int(max_y - min_y)    
                patch_x = int(min_x)
                patch_y = int(wsi_height - max_y)
    
                # New dimensions with a 50% increase (blood model was trained with larger surrounding area)
                new_width = int(roi_width * 1.5)
                new_height = int(roi_height * 1.5)
                adjust_x = (new_width - roi_width) / 2
                adjust_y = (new_height - roi_height) / 2
                new_min_x = max(min_x - adjust_x, 0)  # Ensure we don't go below 0
                new_min_y = max(min_y - adjust_y, 0)  # Ensure we don't go below 0
                new_max_x = min(max_x + adjust_x, wsi_width)  # Ensure we don't exceed image width
                new_max_y = min(max_y + adjust_y, wsi_height)  # Ensure we don't exceed image height
                patch_x = int(new_min_x)
                patch_y = int(wsi_height - new_max_y)  # Flip Y-axis for bottom-left origin
    
                response = cyto_job.get_instance()._get(
                    "{}/{}/window-{}-{}-{}-{}.{}".format("imageinstance", id_image, patch_x, patch_y, new_width, new_height, "png"), {}
                )
    
                if response.status_code in [200, 304] and response.headers['Content-Type'] == 'image/png':
                    image_bytes = BytesIO(response.content)
                    image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
                    im = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    im = cv2.resize(im, (224, 224))
                    arr_out = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)
                    image_bytes.seek(0)                
    
                    compiled_model, output_layer = get_compiled_model_and_output_layer()
                    output_batch = compiled_model([arr_out])[output_layer]
                    softmax_scores = softmax(output_batch[0])
                    max_confidence = np.max(softmax_scores)
                    pred_label = np.argmax(softmax_scores)
                    term_id = label_to_term_id.get(pred_label) if max_confidence >= confidence_threshold else parameters.cytomine_id_cell_term
    
                    if term_id:
                        return Annotation(location=roi_geometry.wkt, id_image=id_image, id_project=project.id, id_terms=[term_id]), pred_label

            except Exception as e:
                print(f"Error processing ROI {roi_id}: {e}")
                return None, None        
        
        for id_image in list_imgs2:
            job.update(status=Job.RUNNING, progress=50, statusComment=f'Processing image: {id_image}')
            # print(f'Processing image: {id_image}')
            imageinfo = ImageInstance(id=id_image, project=parameters.cytomine_id_project)
            imageinfo.fetch()

            # Fetch ROIs for the current image
            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_cell_term],
                project=parameters.cytomine_id_project,
                image=id_image,
                showWKT=True,
                includeAlgo=True,
            )
            roi_annotations.fetch()
            print(roi_annotations)

            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [
                    executor.submit(
                        process_cell, roi, roi.id, roi.location, id_image,
                        confidence_threshold, label_to_term_id, working_path, project, User().fetch(roi.user).algo
                    )
                    for roi in roi_annotations
                ]

                # Gather results and save them
                cytomine_annotations = AnnotationCollection()
                for future in as_completed(futures):
                    annotation, pred_label = future.result()
                    if annotation:
                        cytomine_annotations.append(annotation)
                        if pred_label is not None:
                            class_counters[pred_label] += 1
                    # print(".", end='', flush=True)

                # Save annotations in bulk
                cytomine_annotations.save()

            print(f"Completed processing for image {id_image}")
    
            print("\nClass Counters:")
            for label, count in class_counters.items():
                class_name = class_names[label]
                print(f"Label {label} ({class_name}): {count} predictions")

            end_prediction_time=time.time()

            job.update(status=Job.RUNNING, progress=90, statusComment="Generating scoring for whole-slide image(s)...")
                  
            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            print("Prediction time: ",end_prediction_time-start_prediction_time)

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0;Class 1;Class 2;Total Prediction;Execution Time;Prediction Time\n")
            # f.write("{};{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_c2,pred_total,end_time-start_time,end_prediction_time-start_prediction_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

