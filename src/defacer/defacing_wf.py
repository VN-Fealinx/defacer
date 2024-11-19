from pathlib import Path
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.quickshear import Quickshear
from defacer.mask_creation import gen_mask_wf
import click


@click.command()
@click.option('--indir', type=click.Path(exists=True), help='Folder (directory) containing all the DICOM series (in sub-folders).')
@click.option('--outdir', type=click.Path(exists=False), help='Folder (directory) where the results will be stored.')
@click.option('--threads4mask', default=4, help='Number of CPUs to use for the segmentation of the brain (needed for the defacing).')
@click.option('--modeldir', type=click.Path(exists=True), default='/data/model', help='Folder (directory) where the IA model for the brain masking is stored.')
@click.option('--descriptor', type=click.Path(exists=True), default='/data/model/descriptor.json', help='File (.json) describing the info about the AI model.')
def main(indir: Path, outdir: Path, threads4mask: int, modeldir: Path, descriptor: Path):

    if isinstance(indir, str):
        indir = Path(indir).absolute()
    if isinstance(outdir, str):
        outdir = Path(outdir).absolute()

    print("indir : " + str(indir))

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

    mask_wf = gen_mask_wf(threads4mask, model=modeldir, descriptior=descriptor)
    workflow.add_nodes([mask_wf])

    defacing = Node(Quickshear(), name='defacing')

    workflow.connect(dcm2nii, 'converted_files', mask_wf, 'crunch.img')
    workflow.connect(dcm2nii, 'converted_files', defacing, 'in_file')  # For now, expect 1 file only from dcm2nii ouput, also, the dcm2nii.bids (json) sidecar
    workflow.connect(mask_wf, 'binarize_brain_mask.thresholded', defacing, 'mask_file')

    sink_node = Node(DataSink(),
                     name='sink_node')
    sink_node.inputs.base_directory = str(outdir / 'results')
    sink_node.inputs.substitutions = [
        ('_aquisition_id_', '')
    ]
    workflow.connect(defacing, 'out_file', sink_node, 'defaced_images')
    workflow.connect(dcm2nii, 'bids', sink_node, 'defaced_images.@json')

    workflow.config['execution']['stop_on_first_crash'] = 'True'  # For debug
    workflow.run()


if __name__ == '__main__':
    main()
