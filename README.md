# OSAM
# OSAM: Road Segmentation using MedSAM

## Environment Setup

### 1. Install Segment Anything
To set up the environment for using Segment Anything, follow the steps below:

- **Clone the Segment Anything repository**:
  ```bash
  git clone https://github.com/facebookresearch/segment-anything.git

### 2. Install the requirements.txt

## Model Training

### 1. Download Model Checkpoints:
Model checkpoints are available in the Segment Anything repository. You can choose from different versions of the model, such as: vit_h; vit_l; vit_b 
    
Download the model of your choice and ensure it is stored in an accessible directory.

### 2. Preprocess Training Data:
Use the provided preprocess_train_data.py script to preprocess the training data. This script will convert your dataset into the required format for training.

### 3. Train the Model:
Use the train_froads.py script to train the model. Ensure that you have preprocessed the data before running the training script.

## Road Prediction
     
### 1. Download OSM Road Vector Data:
Download OSM road vector data (in .shp format) from the official OpenStreetMap website. Ensure that the road data is aligned with the coordinate system of the target images and cropped to match the target image size. Save this data in the infer_data/osm/ directory, e.g., xx.shp.
      
### 2. Process Test Data:
    
1. Convert OSM Road Centerline Vector to Raster:
   Use the shp2raster.py script to convert the OSM road centerline vector data into a raster format. The output will be used for point prompt generation.
     ```bash
	python ./osm_process/shp2raster.py

2. Convert OSM Road Centerline Vector to GeoJSON:
   Use the shp2geojson.py script to convert the OSM road centerline vector data into a GeoJSON file for box prompt generation.

     ```bash
	python ./osm_process/shp2geojson.py

3. Dilate the Rasterized Road Data:
   Use the line2dilated.py script to apply dilation on the rasterized road data, creating a road with a certain width (for mask prompt generation).

     ```bash
	python ./osm_process/line2dilated.py
     
## Postprocessing
     
### 1. Fuse Results:
Once you have completed the predictions using point, box, and mask prompts, you can fuse these results using the fuse.py script. This will combine the results from different prompt techniques.

        python fuse.py
      
### 2. Apply Morphological Postprocessing:
Finally, you can apply morphological postprocessing to refine the fused results. Use the post_process.py script to complete the final postprocessing steps.

        python post_process.py
