# XRay2CT-SAM
Leveraging a Fine-tuned Segment Anything Model for Synchronous 3D CT Reconstruction and Tumor Segmentation via a Single X-ray Projection

This is the code repository for XRay2CT-SAM.  

The Reconstruction Toolkit (RTK) is an open source cross-platform software for rapid cone beam CT reconstruction based on the Insight Toolkit (ITK). RTK is an open source package of CBCT reconstruction algorithms, owned by Kitware, and is based on the ITK package extension. RTK implements many existing CT image reconstruction algorithms, including ADMM, SART, SIRT, POCS, etc.  

We use RTK to obtain DRR images from CT images at a certain angle. If you can't properly use the RTK, I recommend that you look through the RTK official website.[RTK - Reconstruction Toolkit (openrtk.org)](https://www.openrtk.org/)  
# ✏️ Introduction
Our network is structured as follows. For more details, please read the paper.

<img width="5172" height="2162" alt="总体框图20250724" src="https://github.com/user-attachments/assets/0d331a20-1aa3-476c-8d06-28917ebda896" />

The S-FI module is shown below.

<img width="2813" height="822" alt="S-FI" src="https://github.com/user-attachments/assets/71f7a659-347a-4b9c-a52e-f7fd6e88c79c" />

The HFE-D module is shown below.

<img width="4107" height="1405" alt="HFE-D" src="https://github.com/user-attachments/assets/13d72454-fc6d-4560-a10c-cd17422845e0" />

# ⚙️ Install  

```
$ git clone https://github.com/wjj1wjj/XRay2CT-SAM.git
$ cd XRay2CT-SAM
$ conda env create -f environment.yaml
$ conda activate XRay2CT-SAM
```

# 💾 Data  

The geometry file is used for the RTK to generate DRR image at a certain angle. The dataset is divided into 3 parts by the TXT file.  
We saved each real 3DCT and corresponding real tumor label into the h5py file and put it into a separate folder according to the 1080 time phase. The structure is as follows:
```
|--h5py
|  |--1
|   |--ct_xray12_label.h5
|  |--2
|    |--ct_xray12_label.h5
   ·········
|--ct_rtk
|  |--1_rtk.mha
|  |--2_rtk.mha
   ·········
```
The reason why we create subfiles under h5py instead of listing all files directly: the generated DRR image will be saved under this path.

# ⏳Train & Test  

To train our model, run:
```
$ python XctNet_MASA/MedSAM/MedXct_train.py
```
To test our model, run:
```
$ python XctNet_MASA/MedSAM/MedXct_test.py
```

# 📊 Contact  
If you have any questions, please feel free to contact bo.liu@buaa.edu.cn.  

# 📜 License  
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.

