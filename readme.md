# Defacing pipeline for brain MRI anonymization

The pipeline takes in DICOM files of head/brain MRI, and outputs corresponding Nifti images with the face removed.

It is composed of the following steps:
1. File retrieval
2. DICOM to Nifti conversion
3. Creation of a brain mask (AI tool)
4. Removal of the face