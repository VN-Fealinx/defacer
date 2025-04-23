from pathlib import Path
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.dcm2nii import Dcm2niix
# from nipype.interfaces.quickshear import Quickshear
from .mask_creation import gen_mask_wf
from .interfaces import Resample_from_to, Nii2dcm, Quickshear2
import click


def as_list(arg, ind):
    if not isinstance(arg, list):
        arg = [arg]
    return arg[ind]


@click.command()
@click.option('--indir', type=click.Path(exists=True), help='Folder (directory) containing all the DICOM series (in sub-folders).')
@click.option('--outdir', type=click.Path(exists=False), help='Folder (directory) where the results will be stored.')
@click.option('--echo_nb', default=1, help='Number of expected images extracted from the DICOM series (default is 1). Must be the same for all series.')
@click.option('--threads4mask', default=4, help='Number of CPUs to use for the segmentation of the brain (needed for the defacing).')
@click.option('--modeldir', type=click.Path(exists=True), default='/data/model', help='Folder (directory) where the IA model for the brain masking is stored.')
@click.option('--descriptor', type=click.Path(exists=True), default='/data/model/brainmask/V0/model_info.json', help='File (.json) describing the info about the AI model.')
@click.option('--dcm_type', default='SVR', help='Expected type of DICOM file (MR or SVR). If not specfied, will create generic DICOM files.')
@click.option('--opening', default=20, help='Number of morphological erosion done during the brain mask cleaning step.')
def main(indir: Path,
         outdir: Path,
         echo_nb: int,
         threads4mask: int,
         modeldir: Path,
         descriptor: Path,
         dcm_type: str = 'SVR',
         opening: int = 20):

    if isinstance(indir, str):
        indir = Path(indir).absolute()
    if isinstance(outdir, str):
        outdir = Path(outdir).absolute()

    print("indir : " + str(indir))

    if echo_nb < 1:
        print('Non-positive echo number. Auto set to 1.')
        echo_nb = 1
    if threads4mask < 1:
        print('Non-positive number of theads. Auto set to 1.')
        threads4mask = 1

    if not outdir.exists():
        Path.mkdir(outdir)

    dcm_folder_list = [dcmdir.stem for dcmdir in indir.iterdir() if dcmdir.is_dir()]

    wf_name = 'defacer_wf'
    workflow = Workflow(wf_name)
    workflow.base_dir = outdir

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
    datagrabber.inputs.base_directory = indir
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

    sink_node = Node(DataSink(),
                     name='sink_node')
    sink_node.inputs.base_directory = str(outdir / 'results')
    sink_node.inputs.substitutions = [
        ('_aquisition_id_', '')
    ]

    mask_wf_list = []
    for echo in range(echo_nb):
        suffix = f'_e{echo + 1}' if echo_nb > 1 else ''
        mask_wf_list.append(gen_mask_wf(threads4mask, model=modeldir, descriptior=descriptor, suffix=suffix, open_iter=opening))
        workflow.add_nodes([mask_wf_list[echo]])

    # To make sure the mask and the original image are in the same space
        correct_affine = Node(Resample_from_to(), name=f"correct_affine{suffix}")
        correct_affine.inputs.spline_order = 0
        correct_affine.inputs.out_suffix = '_affineOK'

        defacing = Node(Quickshear2(), name=f'defacing{suffix}')

        workflow.connect(dcm2nii, ('converted_files', as_list, echo), mask_wf_list[echo], f'crunch{suffix}.img')

        workflow.connect(mask_wf_list[echo], f'binarize_brain_mask{suffix}.thresholded', correct_affine, 'moving_image')
        workflow.connect(dcm2nii, ('converted_files', as_list, echo), correct_affine, 'fixed_image')

        workflow.connect(dcm2nii, ('converted_files', as_list, echo), defacing, 'in_file')  # For now, expect 1 file only from dcm2nii ouput, also, the dcm2nii.bids (json) sidecar
        workflow.connect(correct_affine, 'resampled_image', defacing, 'mask_file')

        nii2dcm = Node(Nii2dcm(), name=f'nii2dcm{suffix}')
        nii2dcm.inputs.out_dir = f'defaced_dicom{suffix}'
        nii2dcm.inputs.dcm_type = dcm_type

        workflow.connect(defacing, 'out_file', nii2dcm, 'nii_input')
        workflow.connect(datagrabber, 'aquisition_folder', nii2dcm, 'dcm_ref')

        workflow.connect(defacing, 'out_file', sink_node, f'defaced_images.@im{suffix}')
        workflow.connect(nii2dcm, 'out_dir', sink_node, f'defaced_images.@dcm{suffix}')
        workflow.connect(dcm2nii, ('bids', as_list, echo), sink_node, f'defaced_images.@json{suffix}')

    workflow.config['execution']['stop_on_first_crash'] = 'True'  # For debug
    result = workflow.run()
    res = {node.itername: node for node in result.nodes}
    out_dcm = []
    for sink in [res[n] for n in res if n.startswith('defacer_wf.sink_node')]:
        out_dcm += [o for o in sink.result.outputs.out_file if 'defaced_dicom' in o]
    return out_dcm


if __name__ == '__main__':
    main()
