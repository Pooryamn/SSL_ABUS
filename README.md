# A multi-task self-supervised approach for mass detection in Automated Breast Ultrasound using Double Attention Recurrent Residual U-Net
======
Breast cancer is the most common and lethal cancer among women worldwide. Early detection using medical imaging technologies can significantly improve treatment outcomes. Automated breast ultrasound, known as ABUS, offers more advantages compared to traditional mammography and has recently gained considerable attention. However, reviewing hundreds of ABUS slices imposes a high workload on radiologists, increasing review time and potentially leading to diagnostic errors. Consequently, there is a strong need for efficient computer-aided detection, CADe, systems. In recent years, researchers have proposed deep learning-based CADe systems to enhance mass detection accuracy. However, these methods are highly dependent on the number of training samples and often struggle to balance detection accuracy with the false positive rate. To reduce the workload for radiologists and achieve high detection sensitivities with low false positive rates, this study introduces a novel CADe system based on a self-supervised framework that leverages unannotated ABUS datasets to improve detection results. The proposed framework is integrated into an innovative 3-D convolutional neural network called DATTR2U-Net, which employs a multi-task learning approach to simultaneously train inpainting and denoising pretext tasks. A fully convolutional network is then attached to the DATTR2U-Net for the detection task. The proposed method is validated on the TDSCABUS public dataset, demonstrating promising detection results with a recall of 0.7963 and a false positive rate of 5.67 per volume that signifies its potential to improve detection accuracy while reducing workload for radiologists.

======
# Authors:
- Poorya MohammadiNasab
- Atousa Khakbaz
- Hamid Behnam
- Ehsan Kozegar
- Mohsen Soryani

====== 
# Supplementary materials:
## [Pre-trained Weights](https://drive.google.com/drive/folders/14XkuninPXx0IlDigjaMjILmb-pzdbxnb?usp=sharing)
## [Sample Dataset](https://drive.google.com/drive/folders/1M_A50q3utWUuO2PY4ugNSMbaa82JEFvE?usp=sharing)

======
# Repository Under Development 🚧  

Thank you for visiting this repository! 🚀  

Please note that this repository is currently a work in progress. While it already contains some essential components, such as the codebase, pretrained weights, and a sample dataset for testing, we are actively working on completing the documentation and guidelines.  

### What's Coming Soon:  
- **Comprehensive Documentation**: Step-by-step instructions to set up the environment and use the code effectively.  
- **Requirements File**: A detailed list of dependencies.  
- **Training & Testing Guidelines**: Clear instructions on how to train models from scratch and test them.  
- **Examples & Tutorials**: Helpful examples to get you started quickly.  

We appreciate your patience and encourage you to check back soon for updates. If you encounter any issues or have suggestions, feel free to open an issue. 😊  

Stay tuned for a fully-documented and user-friendly release! 🚀
