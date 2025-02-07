# Defacing pipeline for brain MRI anonymization
## Presentation

The pipeline takes in DICOM files of head/brain MRI, and outputs corresponding Nifti images with the face removed.

It is composed of the following steps:
1. File retrieval
2. DICOM to Nifti conversion
3. Creation of a brain mask (several steps and an AI tool)
4. Removal of the face

## Building the Apptainer image
Follow these steps
1. Download (or clone) this repository from Github to your machine.
2. Download the model files from here: [https://cloud.efixia.com/sharing/Mn9LB5mIR](https://cloud.efixia.com/sharing/Mn9LB5mIR) and store the zip file (still zipped, no need to uncompress it) in the root folder of the project (i.e. next to the `defacer_apptainer.recipe` file).
3. Make sure you have installed Apptainer on your machine (you will likely need root privileges). The [Quick start page of Apptainer can be found here](https://apptainer.org/docs/user/main/quick_start.html).
4. From the root folder of the project, run `apptainer build defacer.sif defacer_apptainer.recipe`. This will create the apptainer image `defacer.sif` that can be used to run the defacing pipeline.

## Building the Docker image
Follow these steps
1. Download (or clone) this repository from Github to your machine.
2. Download the model files from here: [https://cloud.efixia.com/sharing/Mn9LB5mIR](https://cloud.efixia.com/sharing/Mn9LB5mIR) and store the zip file (still zipped, no need to uncompress it) in the root folder of the project (i.e. next to the `Dockerfile` file).
3. Make sure you have Docker installed on your machine (you will need root privileges).
4. From the root folder of the project, run `docker build -t myId/defacer .` (replace `myId` by your username).

## Running the pipeline through the container
### Input structure
The pipeline expects its input data to be organized in an input folder, containing one sub-folder per aquisition to deface (each aquisition subfolder will here contain the DICOM series of said aquisition). The name of the subfolder will be used to name the output folder.

For example (note that the DICOM files don't need the `.dcm` shown here):

    input_folder
            ├── aquisition_1
            │   ├── xxxxx.dcm
            │   ├── xxxxx.dcm
            │   ├── xxxxx.dcm
            │   :
            │
            ├── aquisition_2
            │   ├── xxxxx.dcm
            │   ├── xxxxx.dcm
            │   ├── xxxxx.dcm
            │   :
            :


> **/!\ Important note**: The pipeline will run on all the aquisitions in the input folder, however they **must** all contain the same number of images (echoes). For example, a SWI acquisition will generally have 2 echoes while a T1w acquisition only 1, so they can't be processed at the same time. The number of echo per DICOM series is spcified by the `--echo_nb` argument (see below).

### Command line
To run the pipeline with the container, use the command line below and:
- replace the `/path/to/input_folder` by your real input folder path
- replace the `/path/to/output_folder` by your real output folder path
- replace `/path/to/defacer.sif` by the actual path to the image
- change the "1" after `--echo_nb` to the number of expected echos (3D images) in each DICOM series
- (opt.) change the "6" after `--threads4mask` by the number of CPUs you want to use for the brain masking step
- Don't touch anything else

Using Apptainer:
```bash
apptainer exec --bind /path/to/input_folder:/mnt/input:rw,/path/to/output_folder:/mnt/output:rw /path/to/defacer.sif deface --indir /mnt/input --outdir /mnt/output --echo_nb 1 --threads4mask 6
```

Using Docker (also replace `myId` here):
```bash
docker docker run --rm --name defacer \
    --volume /path/to/input_folder:/mnt/input_data \
    --volume /path/to/output_folder:/mnt/output \
    myId/defacer \
    deface --indir /mnt/input --outdir /mnt/output --echo_nb 1 --threads4mask 6
```