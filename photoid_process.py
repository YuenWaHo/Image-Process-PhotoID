# Import Libraries
import os
import shutil
import random

from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
import piexif
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap.umap_ as umap
from sklearn.cluster import KMeans
import ultralytics
from ultralytics import YOLO

# Print version information
print("Ultralytics Version:",ultralytics.__version__)
print("OpenCV Version:", cv2.__version__)
print("Piexif Version:", piexif.VERSION)
print("PyTorch Version:", torch.__version__)
print("NumPy Version:", np.__version__)

class ImagePreprocess:
    def __init__(self, study_site, group_id):
        self.study_site = study_site
        self.group_id = group_id

    def extract_exif(self, image_path):
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        if exif_data is not None:
            exif = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                exif[tag_name] = value
            return exif
        return None

    def rename_images(self, image_folder, format_string):
        if not os.path.isdir(image_folder):
            print(f"Error: The directory {image_folder} does not exist.")
            return

        try:
            images = [img for img in os.listdir(image_folder) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
            images.sort()
            renamed_count = 0

            for index, image_name in enumerate(images):
                if image_name.startswith("._"): 
                    continue

                image_path = os.path.join(image_folder, image_name)
                exif = self.extract_exif(image_path)
                date_time = exif.get('DateTime', '').replace(":", "").replace(" ", "_") if exif else "UnknownDate"
                date_time = date_time[0:8]  # Only take the date part (YYYYMMDD)
                frame_number = image_name[-8:-4] 
                image_id = f"{index + 1:04d}"

                new_filename = format_string.format(
                    image_id=image_id,
                    date_time=date_time,
                    STUDY_SITE=self.study_site,
                    GROUP_ID=self.group_id,
                    frame_number=frame_number
                )
                new_image_path = os.path.join(image_folder, new_filename)

                try:
                    os.rename(image_path, new_image_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {image_name} to {new_filename}: {e}")

            print(f"Renamed {renamed_count} images.")

        except Exception as e:
            print(f"An error occurred during the renaming process: {e}")

class ImageAnalysis:
    def __init__(self, input_dir, output_dir=None, model_name="weights/sousa_dorsal_fin.pt"):
        self.input_dir = input_dir   # Only use output_dir if provided, otherwise skip it for single image detection
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'Cropped')
            self.non_cropped_output_dir = os.path.join(output_dir, 'Uncropped')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if not os.path.exists(self.non_cropped_output_dir):
                os.makedirs(self.non_cropped_output_dir)
        else:
            self.output_dir = None
            self.non_cropped_output_dir = None
        
        self.model_name = model_name
    
    def process_images(self):
        model = YOLO(self.model_name)  # Load the specified YOLO model weights
        
        for image_name in os.listdir(self.input_dir):
            if not image_name.startswith("._") and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.input_dir, image_name)
                image = Image.open(image_path)
                exif_data = image.info.get("exif")
                results = model(image_path)

                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box.numpy())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.width, x2), min(image.height, y2)
                        cropped_image = image.crop((x1, y1, x2, y2))
                        suffix = chr(97 + i)  # Append a suffix (a, b, c, etc.)
                        cropped_image_path = os.path.join(self.output_dir, f'{os.path.splitext(image_name)[0]}{suffix}.jpg')
                        cropped_image.save(cropped_image_path, exif=exif_data)
                        print(f'Cropped image saved: {cropped_image_path}')
                else:
                    output_image_path = os.path.join(self.non_cropped_output_dir, image_name)
                    shutil.copy2(image_path, output_image_path)
                    print(f'No objects detected. Original image copied: {output_image_path}')
    
    def detect_single_image(self, image_path):
        """
        Detects objects in a single image and visualizes the results with bounding boxes.
        """
        model = YOLO(self.model_name)
        image = Image.open(image_path)
        results = model(image_path)
        xyxys = results[0].boxes.xyxy

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box in xyxys:
            x1, y1, x2, y2 = box.numpy()
            width = x2 - x1
            height = y2 - y1
            
            x1 -= 0.1 * width
            y1 -= 0.1 * height
            x2 += 0.1 * width
            y2 += 0.1 * height
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

class ImageCluster:
    def __init__(self, input_dir, n_individuals=5):
        self.input_dir = input_dir
        self.output_dir = input_dir
        self.n_individuals = n_individuals
        
        # Set up feature extraction model (ResNet50)
        model = resnet50(weights=ResNet50_Weights.DEFAULT) 
        self.model = torch.nn.Sequential(*list(model.children())[:-1]) 
        self.model.eval()

        # Set up image transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image).flatten().numpy()
        return features
    
    def process_images(self):
        features_list = []
        image_paths = []

        for image_name in os.listdir(self.input_dir):
            if not image_name.startswith("._") and image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.input_dir, image_name)
                features = self.extract_features(image_path)
                features_list.append(features)
                image_paths.append(image_path)

        features_array = np.array(features_list)

        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42, n_jobs=1)
        umap_embedding = reducer.fit_transform(features_array)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=self.n_individuals, random_state=0, n_init='auto')
        kmeans_labels = kmeans.fit_predict(umap_embedding)
        num_clusters = len(set(kmeans_labels))

        for i in range(1, num_clusters + 1):
            cluster_dir = os.path.join(self.input_dir, f'TempID_{i:02}')
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)

        for image_path, label in zip(image_paths, kmeans_labels):
            target_dir = os.path.join(self.input_dir, f'TempID_{label + 1:02}')
            shutil.move(image_path, os.path.join(target_dir, os.path.basename(image_path)))

        print(f'Images have been categorized into {num_clusters} clusters using UMAP and K-Means.')
    
    def display_clusters(self):     # Display 1 image from each cluster folder
        cluster_dirs = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d)) and d.startswith("TempID_")]

        num_clusters = len(cluster_dirs)
        images_per_row = 5
        num_rows = (num_clusters + images_per_row - 1) // images_per_row 

        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 10 * num_rows))
        axes = axes.ravel() 

        for i, cluster_dir_name in enumerate(sorted(cluster_dirs)):
            cluster_dir = os.path.join(self.input_dir, cluster_dir_name)
            image_files = [f for f in os.listdir(cluster_dir) if not f.startswith("._") and f.lower().endswith(('.jpg', '.jpeg', '.png'))] 
            if image_files:
                example_image_path = random.choice(image_files)  
                example_image = Image.open(os.path.join(cluster_dir, example_image_path))
                axes[i].imshow(example_image)
                axes[i].set_title(cluster_dir_name)
            else:
                print(f"No valid images found in {cluster_dir}")
                axes[i].set_title(f'{cluster_dir_name} (No Images)')
                axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()