# Defacing pipeline for brain MRI anonymization

The pipeline takes in DICOM files of head/brain MRI, and outputs corresponding Nifti images with the face removed.

It is composed of the following steps:
1. File retrieval
2. DICOM to Nifti conversion
3. Creation of a brain mask (several steps and an AI tool)
4. Removal of the face

To build the Apptainer image, you will need to download the model files from here: [https://cloud.efixia.com/sharing/Mn9LB5mIR](https://cloud.efixia.com/sharing/Mn9LB5mIR)