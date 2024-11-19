from nipype.pipeline.engine import Workflow, Node
from pathlib import Path
from typing import Union
from defacer.interfaces import (Threshold, Normalization,
                                Conform, Crop, Predict)


def gen_mask_wf(threads: int, model: Union[Path, str], descriptior: Union[Path, str]) -> Workflow:
    workflow = Workflow(name='brain_mask_creation')
    crunch = Node(Conform(),
                  name="crunch")
    crunch.inputs.dimensions = (160, 214, 176)
    crunch.inputs.orientation = 'RAS'

    crunched_normalization = Node(Normalization(percentile=99), name="crunched_normalization")
    workflow.connect(crunch, 'resampled', crunched_normalization, 'input_image')

    node_env = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads)
    }
    brain_mask = Node(Predict(), "brain_mask", environ=node_env)
    brain_mask.inputs.out_filename = 'brain_mask.nii.gz'
    brain_mask.inputs.gpu_number = -1
    brain_mask.plugin_args = {'sbatch_args': f'--nodes 1 --cpus-per-task {threads}'}
    brain_mask.inputs.threads = threads
    brain_mask.inputs.model = model  # To put in the singularity
    brain_mask.inputs.descriptor = descriptior

    workflow.connect(crunched_normalization, 'intensity_normalized', brain_mask, 'img')

    uncrunch_mask = Node(Conform(), name="uncrunch_mask")
    uncrunch_mask.inputs.order = 0
    uncrunch_mask.inputs.ignore_bad_affine = True

    workflow.connect(brain_mask, 'segmentation', uncrunch_mask, 'img')
    workflow.connect(crunch, 'ori_size', uncrunch_mask, 'dimensions')
    workflow.connect(crunch, 'ori_resol', uncrunch_mask, 'voxel_size')
    workflow.connect(crunch, 'ori_orient', uncrunch_mask, 'orientation')

    binarize_brain_mask = Node(Threshold(threshold=0.5), name="binarize_brain_mask")
    binarize_brain_mask.inputs.binarize = True
    binarize_brain_mask.inputs.open = 10  # morphological opening of clusters using a ball of radius 3
    binarize_brain_mask.inputs.minVol = 30000  # Get rif of potential small clusters
    binarize_brain_mask.inputs.clusterCheck = 'size'  # Select biggest cluster

    workflow.connect(uncrunch_mask, 'resampled', binarize_brain_mask, 'img')

    return workflow
