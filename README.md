# 🧱 Miner-Pose Dataset

The **Miner-Pose** dataset is a large-scale dataset designed for human pose estimation in underground mining environments. Unlike existing public datasets which are primarily collected under well-lit, above-ground conditions, **Miner-Pose** is specifically tailored to the unique challenges posed by underground mining scenarios. It includes **12,225 images** collected from real-world mining environments and re-annotated data from the **DsLMF+** dataset, providing valuable data for mining safety and pose estimation research.

---

## 📊 Dataset Overview

- **Total Images**: 12,225
- **Data Source**: Real-world mining footage + reannotations from the DsLMF+ dataset
- **Pose Categories**: Crouching, bending, carrying tools, and other mining behaviors
- **Environment**: Underground mining conditions with occlusion, clutter, and poor lighting

---

## 📥 Dataset Access

> 🔗 **The Miner-Pose dataset is publicly available at the following links:**

- 🇨🇳 **[Baidu Netdisk (国内用户)](https://pan.baidu.com/s/1_otJGyCM1NCT3RBdsO-N0w?pwd=5555)**  
  Access code: `5555`

- 🌍 **Google Drive (for international users)**  
  👉 *(Insert your Google Drive link here)*

> 📝 The dataset is also hosted on GitHub for accessing code and annotations.  
> Please refer to the license section below for terms of use and redistribution.

---

## 📁 Dataset Content

- **Images**: 12,225 JPG images extracted from surveillance videos (2021–2024)
- **Annotations**: Human body keypoints and bounding boxes (COCO keypoint format)
- **Format**: JPG images and JSON annotations

---

## 📌 Comparison with Existing Datasets

| Year | Dataset Name | Data Scale       | Public |
|------|--------------|------------------|--------|
| 2023 | -            | -                | No     |
| 2024 | Colliery-1   | 600 video clips  | Partial |
| 2024 | -            | 5916 images      | No     |
| 2025 | -            | -                | No     |
| 2025 | -            | 5808 images      | No     |
| **2025** | **Miner-Pose** | **12,225 images** | **Yes** |

---

## 🛠️ Data Collection and Annotation Process

![Annotation Process](https://github.com/ITVR-lab/Miner-Pose/blob/main/process.png)

### 📹 Miner-Pose Dataset Creation Steps

1. **Data Collection**: Our team recorded 400 surveillance video clips (2021–2024) in underground mines, contributing over 4,000 original images to the dataset, in addition to 7,986 re-annotated images from DsLMF+.
2. **Data Cleaning**: Removal of redundant/invalid frames and resizing  
3. **Initial Annotation**: Auto-labeling via X-AnyLabeling with COCO format  
4. **Manual Refinement**: Final corrections using Labelme

---

## 🧪 Dataset Split

- **Training Set**: 9,780 images (80%)
- **Validation Set**: 2,445 images (20%)

---

## ✅ Usage Guidelines

- **Intended Use**: For developing pose estimation models in mining safety applications
- **Ethical Considerations**: Data collected under privacy-respecting and ethical protocols. Please ensure legal and ethical compliance in further use.

---

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
You are free to share and adapt the dataset, with appropriate credit.

---

## 📚 Citation

If you use the **Miner-Pose** dataset in your research, please cite:

> [ Efficient Human Pose Estimation in Complex Coal Mining Scenes via Keypoint Partitioning Adaptive Convolution]-in review

> 📝 **Acknowledgment**:  
> We gratefully acknowledge the creators of the **DsLMF+** dataset, which served as the source for 7,986 images in our dataset.  
> The original DsLMF+ dataset and its publication are available at:  
> 🔗 **https://doi.org/10.1038/s41597-023-02322-9**


