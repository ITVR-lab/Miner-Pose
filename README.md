# Miner-Pose Dataset

The **Miner-Pose** dataset is a large-scale dataset designed for human pose estimation in underground mining environments. Unlike existing public datasets which are primarily collected under well-lit, above-ground conditions, **Miner-Pose** is specifically tailored to the unique challenges posed by underground mining scenarios. It includes over 12,225 images collected from real-world mining environments and re-annotated data from the **DsLMF+** dataset, providing valuable data for mining safety and pose estimation research.

![Miner-Pose Overview](https://github.com/your-username/Miner-Pose/images/overview.png)

## Dataset Overview

- **Total Images**: 12,225 images
- **Data Source**: Newly collected real-world images and reannotations from the DsLMF+ dataset.
- **Pose Categories**: Various miner poses including crouching, bending, carrying tools, and other behaviors specific to mining tasks.
- **Environment**: Underground mining settings with dense, cluttered, and occluded environments.

## Dataset Access

The **Miner-Pose** dataset is publicly available at:

[Insert Dataset Link or DOI here]

The dataset is hosted on GitHub and can be downloaded in full through the link above. Please refer to the license for terms of use and redistribution.

## Dataset Content

The dataset contains:

- **Images**: Over 12,225 images collected from surveillance video clips in mining environments between 2021 and 2024.
- **Annotations**: Body keypoints and bounding boxes for miner poses, formatted according to the COCO keypoint annotation standard.
- **Format**: Annotations are provided in JSON format, and images are in JPG format.

## Comparison to Other Datasets

The **Miner-Pose** dataset fills a significant gap in the available datasets for human pose estimation in mining environments. Table 1 compares the **Miner-Pose** dataset with other human pose estimation datasets created for mine scenes.

| Year | Dataset Name | Data Scale  | Public |
|------|--------------|-------------|--------|
| 2023 | -            | -           | No     |
| 2024 | Colliery-1   | 600 video clips | Partial |
| 2024 | -            | 5916 images | No     |
| 2025 | -            | -           | No     |
| 2025 | -            | 5808 images | No     |
| **2025** | **Miner-Pose** | **12,225 images** | **Yes** |

## Data Collection and Preprocessing

### Re-annotation of DsLMF+ Data

The **DsLMF+** dataset was originally created for object detection tasks in underground coal mines, containing 138,004 images annotated with categories such as personnel, hydraulic shields, coal blocks, and more. For our dataset, we selected 7,986 images from **DsLMF+**, which contained annotations related to miner behaviors. These images were re-annotated with detailed human pose information, using the **COCO** keypoint annotation format to ensure compatibility with standard pose estimation benchmarks.

![DsLMF+ Re-annotation Process](https://github.com/ITVR-lab/Miner-Pose/process.png)

### Miner-Pose Dataset Creation

Our **Miner-Pose** dataset was created through a systematic process, which includes the following steps:

1. **Data Collection**: We collected 400 surveillance video clips from various critical areas within underground mines. These clips, recorded between 2021 and 2024, were processed to extract frames in JPG format.
   
2. **Data Cleaning**: The extracted images were cleaned to remove redundant or invalid samples, and all retained images were resized to a consistent resolution for easier annotation.

3. **Initial Annotations**: Using the **X-AnyLabeling** tool, we pre-annotated the images with human pose keypoints based on the **COCO** format. This step improved labeling efficiency.

4. **Manual Refinement**: To ensure accuracy, we manually refined the keypoints using the **Labelme** tool.

As a result, the **Miner-Pose** dataset contains high-quality, diverse images suitable for human pose estimation in mining scenarios.

## Dataset Split

The dataset is randomly divided into training and validation sets in an 80:20 ratio:

- **Training Set**: 9,780 images
- **Validation Set**: 2,445 images

## Usage Guidelines

- **Intended Use**: This dataset is primarily intended for developing and benchmarking human pose estimation models in the context of underground mining safety and monitoring.
- **Ethical Considerations**: The dataset was collected following ethical guidelines to ensure the safety and privacy of the miners involved in the recordings. Users are encouraged to ensure compliance with all relevant laws and regulations when utilizing the dataset.

## Citation

If you use the **Miner-Pose** dataset in your research, please cite it as follows:

