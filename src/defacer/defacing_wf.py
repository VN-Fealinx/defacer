from pathlib import Path
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.quickshear import Quickshear
from defacer.mask_creation import gen_mask_wf


def main(inDir: Path, outDir: Path, threads4mask: int):

    if isinstance(inDir, str):
        inDir = Path(inDir)
    if isinstance(outDir, str):
        outDir = Path(outDir)

    dcm_folder_list = [dcmdir.stem for dcmdir in inDir.iterdir() if dcmdir.is_dir()]

    wf_name = 'defacer_wf'
    workflow = Workflow(wf_name)
    workflow.base_dir = outDir

    folder_iterator = Node(
        IdentityInterface(
            fields=['aquisition_id'],
            mandatory_inputs=True),
        name="folder_iterator")
    folder_iterator.iterables = ('aquisition_id', dcm_folder_list)

    datagrabber = Node(DataGrabber(  # placeholder for now
        infields=['aquisition_id'],
        outfields=['aquisition_folder']),
        name='datagrabber')
    datagrabber.inputs.base_directory = inDir
    datagrabber.inputs.raise_on_empty = True
    datagrabber.inputs.sort_filelist = True
    datagrabber.inputs.template = '%s'
    # datagrabber.inputs.field_template = {acq[0]: f'%s/{acq[1]}/{files}' for acq in acquisitions}
    # datagrabber.inputs.template_args = {acq[0]: [['subject_id']] for acq in acquisitions}

    workflow.connect(folder_iterator, 'aquisition_id', datagrabber, 'aquisition_id')

    dcm2nii = Node(Dcm2niix(), name='dcm2nii')
    dcm2nii.inputs.anon_bids = True
    dcm2nii.inputs.out_filename = 'converted_%p'

    workflow.connect(datagrabber, 'aquisition_folder', dcm2nii, 'source_dir')

    mask_wf = gen_mask_wf(threads4mask)
    workflow.add_nodes([mask_wf])

    defacing = Node(Quickshear(), name='defacing')

    workflow.connect(dcm2nii, 'converted_files', mask_wf, 'preconform.img')
    workflow.connect(dcm2nii, 'converted_files', defacing, 'in_file')  # For now, expect 1 file only from dcm2nii ouput, also, the dcm2nii.bids (json) sidecar
    workflow.connect(mask_wf, 'binarize_brain_mask.thresholded', defacing, 'mask_file')
