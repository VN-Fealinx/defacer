from nipype.pipeline.engine import Workflow, Node
from pathlib import Path
from typing import Union
from defacer.interfaces import (Threshold, Normalization,
                                Conform, Predict)


def gen_mask_wf(threads: int, model: Union[Path, str], descriptor: Union[Path, str], suffix: str, open_iter=20) -> Workflow:
    workflow = Workflow(name=f'brain_mask_creation{suffix}')
    crunch = Node(Conform(),
                  name=f"crunch{suffix}")
    crunch.inputs.dimensions = (160, 214, 176)
    crunch.inputs.orientation = 'RAS'
    crunch.inputs.ignore_bad_affine = True

    crunched_normalization = Node(Normalization(percentile=99),
                                  name=f"crunched_normalization{suffix}")
    workflow.connect(crunch, 'resampled', crunched_normalization, 'input_image')

    node_env = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads)
    }
    brain_mask = Node(Predict(),
                      name=f"brain_mask{suffix}", environ=node_env)
    brain_mask.inputs.out_filename = 'brain_mask.nii.gz'
    brain_mask.inputs.gpu_number = -1
    brain_mask.plugin_args = {'sbatch_args': f'--nodes 1 --cpus-per-task {threads}'}
    brain_mask.inputs.threads = threads
    brain_mask.inputs.model = model  # To put in the singularity
    brain_mask.inputs.descriptor = descriptor

    workflow.connect(crunched_normalization, 'intensity_normalized', brain_mask, 'img')

    uncrunch_mask = Node(Conform(),
                         name=f"uncrunch_mask{suffix}")
    uncrunch_mask.inputs.order = 0
    uncrunch_mask.inputs.ignore_bad_affine = True

    workflow.connect(brain_mask, 'segmentation', uncrunch_mask, 'img')
    workflow.connect(crunch, 'ori_size', uncrunch_mask, 'dimensions')
    workflow.connect(crunch, 'ori_resol', uncrunch_mask, 'voxel_size')
    workflow.connect(crunch, 'ori_orient', uncrunch_mask, 'orientation')

    binarize_brain_mask = Node(Threshold(threshold=0.5),
                               name=f"binarize_brain_mask{suffix}")
    binarize_brain_mask.inputs.binarize = True
    binarize_brain_mask.inputs.open_iter = open_iter
    binarize_brain_mask.inputs.minVol = 30000  # Get rif of potential small clusters
    binarize_brain_mask.inputs.clusterCheck = 'size'  # Select biggest cluster

    workflow.connect(uncrunch_mask, 'resampled', binarize_brain_mask, 'img')

    return workflow
