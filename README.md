# 🧱 Miner-Pose Dataset

The **Miner-Pose** dataset is a large-scale dataset designed for human pose estimation in underground mining environments. Unlike existing public datasets, which are primarily collected under well-lit, above-ground conditions, **Miner-Pose** is specifically tailored to the unique challenges posed by underground mining scenarios. It includes **12,225 images** collected from real-world mining environments and re-annotated data from the **DsLMF+** dataset, providing valuable data for mining safety and pose estimation research.

---

## 📊 Dataset Overview

- **Total Images**: 12,225
- **Data Source**: Real-world mining footage + re-annotations from the DsLMF+ dataset
- **Pose Categories**: Crouching, bending, carrying tools, and other mining behaviors
- **Environment**: Underground mining conditions with occlusion, clutter, and poor lighting

---

## 📥 Dataset Access

> [!IMPORTANT]
> **A completed application form is required for every access request.**
>
> We have received a large number of incomplete, automated, duplicate, and bulk access requests, including spam. These requests are difficult to review and have significantly increased processing time. To help us verify legitimate academic and research use, all applicants must complete the repository's [Dataset Access Application Form](./DATASET_ACCESS_APPLICATION.md) before requesting access.

### How to Apply

1. Download and complete the [Dataset Access Application Form](./DATASET_ACCESS_APPLICATION.md).
2. Submit the completed form through the private contact or submission channel designated by the repository maintainers.
3. Request access to the dataset through Google Drive:
   - 🌍 **Google Drive**: https://drive.google.com/file/d/1YM_hInwtQthsSDY4ZMvyMDgTIldpZtiy/view?usp=sharing

Please note:

- A Google Drive access request **without a completed application form will not be processed**.
- Each form must represent **one individual applicant and one clearly described research project**.
- Incomplete, unverifiable, automated, duplicate, or bulk applications may be rejected without further review.
- Applicants should provide an institutional email address whenever possible.
- Please do not publish personal information in a public GitHub issue. Use only the private submission channel designated by the maintainers.
- Due to the volume of requests, review may take additional time. Please avoid sending repeated follow-up requests.

By requesting or accessing the dataset, applicants agree to comply with the dataset license, citation requirements, and all applicable legal and ethical obligations.

> 📝 The repository also provides the associated code and annotations. Please refer to the license section below for the terms of use and redistribution.

---

## 📁 Dataset Content

- **Images**: 12,225 JPG images extracted from surveillance videos (2021–2024)
- **Annotations**: Human body keypoints and bounding boxes (COCO keypoint format)
- **Format**: JPG images and JSON annotations

---

## 📌 Comparison with Existing Datasets

| Year | Dataset Name | Data Scale | Public |
|------|--------------|------------|--------|
| 2023 | - | - | No |
| 2024 | Colliery-1 | 600 video clips | Partial |
| 2024 | - | 5916 images | No |
| 2025 | - | - | No |
| 2025 | - | 5808 images | No |
| **2025** | **Miner-Pose** | **12,225 images** | **Yes** |

---

## 🛠️ Data Collection and Annotation Process

![Annotation Process](https://github.com/ITVR-lab/Miner-Pose/blob/main/process.png)

### 📹 Miner-Pose Dataset Creation Steps

1. **Data Collection**: Our team recorded 400 surveillance video clips (2021–2024) in underground mines, contributing over 4,000 original images to the dataset, in addition to 7,986 re-annotated images from DsLMF+.
2. **Data Cleaning**: Removal of redundant or invalid frames and image resizing
3. **Initial Annotation**: Auto-labeling via X-AnyLabeling in COCO format
4. **Manual Refinement**: Final corrections using Labelme

---

## 🧪 Dataset Split

- **Training Set**: 9,780 images (80%)
- **Validation Set**: 2,445 images (20%)

---

## ✅ Usage Guidelines

- **Intended Use**: Developing pose estimation models for mining safety applications
- **Ethical Considerations**: The data were collected under privacy-respecting and ethical protocols. Users must ensure legal and ethical compliance in any further use.

---

## 📄 License

This dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
You are free to share and adapt the dataset with appropriate credit.

---

## 📚 Citation

If you use the **Miner-Pose** dataset in your research, please cite:

```bibtex
@article{wu2026efficient,
  title={Efficient human pose estimation in complex coal mining scenes via Keypoint Partitioning Adaptive Convolution},
  author={Wu, Jin and Zhou, Huaping and Meng, Xiangrui and Wu, Tao},
  journal={Alexandria Engineering Journal},
  volume={137},
  pages={312--328},
  year={2026},
  publisher={Elsevier}
}
```

> 📝 **Acknowledgment**
>
> We gratefully acknowledge the creators of the **DsLMF+** dataset, which served as the source for 7,986 images in our dataset.
>
> The original DsLMF+ dataset and its publication are available at:
>
> 🔗 https://doi.org/10.1038/s41597-023-02322-9
