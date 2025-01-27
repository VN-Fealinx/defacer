from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import isdefined
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    traits, TraitedSpec, CommandLine)
import os
import warnings
import numpy as np
from pathlib import Path

from typing import Tuple
from copy import deepcopy
from skimage.measure import label
from skimage.morphology import opening, binary_erosion, binary_dilation, ball

import nibabel.processing as nip
import nibabel as nib
from scipy import ndimage
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from functools import reduce

# %% Functions used by the interfaces


def histogram(array, percentile, bins):
    """Create an histogram with a numpy array. Retrieves the largest value in
    the first axis of of the histogram and returns the corresponding value on
    the 2nd axe (that of the voxel intensity value) between two bounds of the
    histogram to be defined

    Args:
        array (array): histogram with 2 axes: one for the number of voxels and
                       the other for the intensity value of the voxels
        bins (int): number of batchs to gather data

    Returns:
        mode (int): most frequent value of histogram
    """
    x = array.reshape(-1)
    hist, edges = np.histogram(x, bins=bins)

    mode = get_mode(hist, edges, bins)

    return mode


def get_mode(hist: np.array,
             edges: np.array,
             bins: int):
    """Get most frequent value in an numpy histogram composed by
    frequence (hist) and values (edges)

    Args:
        hist (np.array): frequence of values
        edges (np.array): different values possible in histogram
        bins (int): number of batchs

    Returns:
        mode (int): most frequent value
    """
    inf = int(0.2 * bins)
    sup = int(0.9 * bins)
    index = np.where(hist[inf:sup] == hist[inf:sup].max())
    mode = edges[inf+index[0]][0]

    return mode


def fisin(arr, vals):
    '''
    Fast np.isin function using reccursive bitwise_or function
    (here represented with the lambda function, because slightly faster(?))
    '''
    try:
        arrl = [arr == val for val in vals]
    except TypeError:  # Typically if there is only 1 value
        arrl = [arr == vals]
    return reduce(lambda x, y: x | y, arrl)


def normalization(img: nib.Nifti1Image,
                  percentile: int,
                  brain_mask: nib.Nifti1Image = None,
                  inverse: bool = False) -> nib.Nifti1Image:
    """We remove values above the 99th percentile to avoid hot spots,
       set values below 0 to 0, set values above 1.3 to 1.3 and normalize
       the data between 0 and 1.

       Args:
           img (nib.Nifti1Image): image to process
           percentile (int): value to threshold above this percentile
           brain_mask (nib.Nifti1Image): Brain mask
           inverse (bool): Wether to "inverse" the resulting image (1-val for all voxels in the brain). Requires a brain mask.

       Returns:
        nib.Nifti1Image: normalized image
    """
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")

    if inverse and not brain_mask:
        raise ValueError('No brain mask was provided while the "inverse" option was selected.')

    # We suppress values above the 99th percentile to avoid hot spots
    array = np.nan_to_num(img.get_fdata())
    print(np.max(array))
    array[array < 0] = 0
    # calculate percentile
    if not brain_mask:
        value_percentile = np.percentile(array, percentile)
    else:
        brain_mask_array = np.squeeze(brain_mask.get_fdata())
        value_percentile = np.percentile(array[np.squeeze(brain_mask_array) != 0], percentile)

    # scaling the array with the percentile value
    array /= value_percentile

    # anything values less than 0 are set to 0 and we set to 1.3 the values greater than 1.3
    array[array > 1.3] = 1.3

    # We normalize the data between 0 and 1
    array_normalized = array / 1.3

    # Inversion (usually for T2w) if needed
    if inverse:
        array_normalized[brain_mask_array.astype(bool)] = 1 - array_normalized[brain_mask_array.astype(bool)]

    # Normalization information
    x = array_normalized.reshape(-1)
    bins = 64
    hist, edges = np.histogram(x, bins=bins)

    mode = get_mode(hist, edges, bins)

    img_nifti_normalized = nip.Nifti1Image(array_normalized.astype('f'), img.affine)

    return img_nifti_normalized, mode


def threshold(img: nib.Nifti1Image,
              thr: float = 0.4,
              sign: str = '+',
              binarize: bool = False,
              open_iter: int = 0,
              clusterCheck: str = 'size',
              minVol: int = 0) -> nib.Nifti1Image:
    """Create a brain_mask by putting all the values below the threshold.
       to 0. Offer filtering options if multiple clusters are expected.

       Args:
            img (nib.Nifti1Image): image to process
            thr (float): appropriate value to select mostly brain tissue
                (white matter) and remove background
            sign (str): '+' zero anything below, '-' zero anythin above threshold
            binarize (bool): make a binary mask
            open_iter (int): do a morphological opening using the given int for the radius
                of the ball used as footprint. If 0 is given, skip this step.
            clusterCheck (str): Can be 'top', 'size', or 'all'. Labels the clusters in the mask,
                then keep the one highest in the brain if 'top' was selected, or keep
                the biggest cluster if 'size' was selected (but will raise an error if
                it's not the one at the top). 'all' doesn't do any check.
            minVol (int): Removes clusters with volume under the specified value. Should
                be used if clusterCheck = 'top'

       Returns:
           nib.Nifti1Image: preprocessed image
    """
    import numpy as np
    import nibabel as nib

    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise TypeError("Only Nifti images are supported")
    if not isinstance(thr, float):
        raise TypeError("'thr' must be a float")
    if not clusterCheck in ('top', 'size', 'all'):
        raise ValueError(
            f"Input for clusterCheck should be 'top', 'size' or 'all' but {clusterCheck} was given.")

    array = img.get_fdata().squeeze()
    if sign == '+' and not binarize:
        array[array < thr] = 0
    elif sign == '+' and binarize:
        array = array > thr
        array = array.astype(np.uint8)
    elif sign == '-' and not binarize:
        array[array > thr] = 0
    elif sign == '-' and binarize:
        array = array < thr
        array = array.astype(np.uint8)
    else:
        raise ValueError(f'Unsupported sign argument value {sign} (+ or -)...')

    if open_iter:
        if binarize:
            for i in range(open_iter):
                array = binary_erosion(array)
            for i in range(open_iter):
                array = binary_dilation(array)
            array = array.astype(np.uint8)
        else:
            array = opening(array, footprint=ball(open_iter))
    if clusterCheck in ('top', 'size') or minVol:
        labeled_clusters = label(array)
        clst,  clst_cnt = np.unique(
            labeled_clusters[labeled_clusters > 0],
            return_counts=True)
        # Sorting the clusters by size
        sort_ind = np.argsort(clst_cnt)[::-1]
        clst,  clst_cnt = clst[sort_ind],  clst_cnt[sort_ind]
        if minVol:
            clst = clst[clst_cnt > minVol]
        if clusterCheck in ('top', 'size'):
            maxInd = []
            for c in clst:
                zmax = np.where(labeled_clusters == c)[2].max()
                maxInd.append(zmax)
            topClst = clst[np.argmax(maxInd)]  # Highest (z-axis) cluster
            if clusterCheck == 'top':
                cluster_mask = (labeled_clusters == topClst)
            else:
                if not topClst == clst[0]:
                    raise ValueError(
                        'The biggest cluster in the mask is not the one at '
                        'the top of the brain. Check the data for that participant.')
                cluster_mask = (labeled_clusters == clst[0])
        else:  # only minVol filtering
            cluster_mask = fisin(labeled_clusters, clst)
        array *= cluster_mask

    thresholded = nip.Nifti1Image(array.astype('f'), img.affine)

    return thresholded


def crop(roi_mask: nib.Nifti1Image,
         apply_to: nib.Nifti1Image,
         dimensions: Tuple[int, int, int],
         cdg_ijk: np.ndarray = None,
         default: str = 'ijk',
         safety_marger: int = 5
         ) -> Tuple[nib.Nifti1Image,
                    Tuple[int, int, int],
                    Tuple[int, int, int],
                    Tuple[int, int, int]]:
    """Adjust the real-world referential and crop image.

    If a mask is supplied, the procedure uses the center of mass of the mask as a crop center.

    If no mask is supplied, and default is set to 'xyz' the procedure computes the ijk coordiantes of the affine
    referential coordiantes origin. If set to 'ijk', the middle of the image is used.

    Args:
        roi_mask (nib.Nifti1Image): mask used to define the center
                                   of the bounding box (center of gravity of mask)
        apply_to (nib.Nifti1Image): image to crop
        dimensions (Tuple[int, int, int], optional): volume dimensions.
                                                     Defaults to (256 , 256 , 256).
        cdg_ijk: arbitrary crop center ijk coordinates
        safety_marger (int): added deviation from the top of the image if the brain mask is offset

    Returns:
        nib.Nifti1Image: preprocessed image
        crop center ijk coordiantes
        bouding box top left ijk coordiantes
        bounding box bottom right coordinates
    """
    start_ornt = io_orientation(apply_to.affine)
    end_ornt = axcodes2ornt("RAS")
    transform = ornt_transform(start_ornt, end_ornt)

    # Reorient first to ensure shape matches expectations
    apply_to = apply_to.as_reoriented(transform)
    if not isinstance(apply_to, nib.nifti1.Nifti1Image):
        raise TypeError("apply_to: only Nifti images are supported")

    if roi_mask and not isinstance(roi_mask, nib.nifti1.Nifti1Image):
        raise TypeError("roi_mask: only Nifti images are supported")
    elif not roi_mask and not cdg_ijk:
        if default == 'xyz':
            # get cropping center from xyz origin
            cdg_ijk = np.linalg.inv(apply_to.affine) @ np.array([0.0, 0.0, 0.0, 1.0])
            cdg_ijk = np.ceil(cdg_ijk).astype(int)[:3]
        elif default == "ijk":
            cdg_ijk = np.ceil(np.array(apply_to.shape) / 2).astype(int)
        else:
            raise ValueError(f"argument 'default' value {default} not valid")
    elif roi_mask and not cdg_ijk:
        # get CoG from mask as center
        start_ornt = io_orientation(roi_mask.affine)
        end_ornt = axcodes2ornt("RAS")
        transform = ornt_transform(start_ornt, end_ornt)

        # Reorient first to ensure shape matches expectations
        roi_mask = roi_mask.as_reoriented(transform)
        required_ndim = 3
        if roi_mask.ndim != required_ndim:
            raise ValueError("Only 3D images are supported.")
        if len(dimensions) != required_ndim:
            raise ValueError(f"`dimensions` must have {required_ndim} values")
        cdg_ijk = np.ceil(np.array(
            ndimage.center_of_mass(
                roi_mask.get_fdata().astype(bool)))).astype(int)

    # Calculation of the center of gravity of the mask, we round and convert
    # to integers

    # We will center the block on the center of gravity, so we cut the size in
    # 2
    halfs = np.array(dimensions)/2
    # we need integers because it is used for indices of the "array of voxels"
    halfs = halfs.astype(int)

    # the ijk of the lowest voxel in the box
    bbox1 = cdg_ijk - halfs
    # the highest ijk voxel of the bounding box
    bbox2 = halfs + cdg_ijk

    array_out = np.zeros(dimensions, dtype=apply_to.header.get_data_dtype())
    print(f"bbox1: {bbox1}")
    print(f"bbox2: {bbox2}")
    print(f"cdg_ijk: {cdg_ijk}")
    offset_ijk = abs(bbox1) * abs(np.uint8(bbox1 < 0))
    bbox1[bbox1 < 0] = 0
    for i in range(3):
        if bbox2[i] > apply_to.shape[i]:
            bbox2[i] = apply_to.shape[i]
    span = bbox2 - bbox1
    print(f"span: {span}")
    print(f"offset: {offset_ijk}")

    if roi_mask:
        vec = np.sum(roi_mask.get_fdata().astype(bool), axis=(0, 1))
        top_mask_slice_index = np.where(np.squeeze(vec != 0))[0].tolist()[-1]

        if bbox2[2] <= top_mask_slice_index:

            # we are too low, we nned to move the crop box up
            # (because brain mask is wrong and includes stuff in the neck and shoulders)

            delta = top_mask_slice_index - bbox2[2] + safety_marger
            bbox1[2] = bbox1[2] + delta
            bbox2[2] = bbox2[2] + delta
            cdg_ijk[2] = cdg_ijk[2] + delta
            print(f"reworked bbox1: {bbox1}")
            print(f"reworked bbox2: {bbox2}")

    array_out[offset_ijk[0]:offset_ijk[0] + span[0],
              offset_ijk[1]:offset_ijk[1] + span[1],
              offset_ijk[2]:offset_ijk[2] + span[2]] = apply_to.get_fdata()[
        bbox1[0]:bbox2[0],
        bbox1[1]:bbox2[1],
        bbox1[2]:bbox2[2]]

    # We correct the coordinates, so first we have to convert ijk to xyz for
    # half block size and centroid
    cdg_xyz = apply_to.affine @ np.append(cdg_ijk, 1)
    halfs_xyz = apply_to.affine @ np.append(cdg_ijk - bbox1, 1)
    padding_xyz = apply_to.affine @ np.append(tuple(offset_ijk), 1)
    offset_padding = apply_to.affine[:, 3] - padding_xyz
    print(f"padding: {padding_xyz}")
    print(f"padding offset: {offset_padding}")
    print(f"halfs xyz: {halfs_xyz}")

    # on recopie la matrice affine de l'image, on va la modifier
    affine_out = deepcopy(apply_to.affine)

    # We shift the center of the image reference frame because we start from
    # less far (image is smaller)
    # And the center is no longer quite the same

    affine_out[:, 3] = affine_out[:, 3] + (cdg_xyz - halfs_xyz) + offset_padding

    # We write the result image
    cropped = nip.Nifti1Image(array_out.astype('f'), affine_out)

    return cropped, cdg_ijk, bbox1, bbox2


# %% Interfaces

class ConformInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply conform function on the
    image """
    img = traits.File(exists=True, desc='NIfTI formated input image to conform to agiven shape',
                      mandatory=True)

    dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                              default=(256, 256, 256),
                              usedefault=True,
                              desc='The minimal array dimensions for the'
                              'intermediate conformed image.')

    order = traits.Int(3, desc="Order of spline interpolation", usedefault=True)

    voxel_size = traits.Tuple(traits.Float, traits.Float, traits.Float,
                              desc='resampled voxel size',
                              mandatory=False)

    orientation = traits.Enum('RAS', 'LAS',
                              'RPS', 'LPS',
                              'RAI', 'LPI',
                              'RPI', 'LAP',
                              'RAP',
                              desc="orientation of image volume brain",
                              usedefault=True)

    ignore_bad_affine = traits.Bool(False,
                                    mandatory=False,
                                    usedefault=True,
                                    desc='If True, does not check if the affine is correct')


class ConformOutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nib.Nifti1Image): transformed image
    """
    resampled = traits.File(exists=True,
                            desc='Image conformed to the required voxel size and shape.')

    ori_size = traits.Tuple(traits.Int, traits.Int, traits.Int, desc='Size of the original input img')
    ori_resol = traits.Tuple(traits.Float, traits.Float, traits.Float, desc='Resolution of the original input img')
    ori_orient = traits.String(desc='Orientation of the original input img')

    corrected_affine = traits.Any(desc=('If the conformed image had a bad affine matrix that needed to be '
                                        'corrected before the conformation, this output contains the corrected affine '
                                        'as a 2D Numpy array. Otherwise it stays undefined'))


class Conform(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image):
            NIfTI image file to process
            dimensions (int, int, int): minimal dimension
            order (int): Order of spline interpolation
            voxel_size (float, float, float): Voxel size of final image
            orientation (string): orientation of the volume brain
        output_spec (nib.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            conform image to desired voxel sizes and dimensions
    """
    input_spec = ConformInputSpec
    output_spec = ConformOutputSpec

    def _run_interface(self, runtime):
        """Run main programm
        Return: runtime
        """
        fname = self.inputs.img

        img = nib.funcs.squeeze_image(nib.load(fname))  # type: nib.Nifti1Image
        ori_size = img.shape[:3]
        ori_resol = tuple(img.header['pixdim'][1:4])
        ori_orient = ''.join(nib.aff2axcodes(img.affine))

        setattr(self, 'ori_size', ori_size)
        setattr(self, 'ori_resol', ori_resol)
        setattr(self, 'ori_orient', ori_orient)

        simplified_affine_centered = None

        if not (isdefined(self.inputs.voxel_size)):
            # resample so as to keep FOV
            voxel_size = np.divide(np.multiply(img.header['dim'][1:4], img.header['pixdim'][1:4]).astype(np.double),
                                   self.inputs.dimensions)
        else:
            voxel_size = self.inputs.voxel_size

        if not self.inputs.ignore_bad_affine:
            # Create new affine (no rotation, centered on center of mass) if the affine is corrupted
            rot, trans = nib.affines.to_matvec(img.affine)
            rot_norm = rot.dot(np.diag(1/img.header['pixdim'][1:4]))  # putting the poration in isotropic space
            test1 = np.isclose(rot_norm.dot(rot_norm.T), np.eye(3), atol=0.0001).all()  # rot x rot.T must give an indentity matrix
            test2 = np.isclose(np.abs(np.linalg.det(rot_norm)), 1, atol=0.0001)  # Determinant for the rotation must be 1
            if not all([test1, test2]):
                warn_msg = (
                    f"BAD AFFINE: in {fname}\n"
                    "The image's affine is corrupted (not encoding a proper rotation).\n"
                    "To avoid problems during registration, a new affine was createdusing the center of mass as origin and "
                    "ignoring any rotation specified by the affine (but keeping voxel dim and left/right orientation).\n"
                    "This will misalign the masks (brain masks and cSVD biomarkers) compared to the raw images but will not "
                    "be a problem if you use the intensity normalized images from the img_preproc folder of the results."
                )
                warnings.warn(warn_msg)
                vol = img.get_fdata()
                cdg_ijk = np.round(ndimage.center_of_mass(vol))
                # As the affine may be corrupted, we discard it and create a simplified version (without rotations)
                simplified_rot = np.eye(3) * img.header['pixdim'][1:4]  # Keeping the voxel dimensions
                simplified_rot[0] *= img.header['pixdim'][0]  # Keeping the L/R orientation
                trans_centered = -simplified_rot.dot(cdg_ijk)
                simplified_affine_centered = nib.affines.from_matvec(simplified_rot, trans_centered)
                img = nib.Nifti1Image(vol.astype('f'), simplified_affine_centered)
        setattr(self, 'corrected_affine', simplified_affine_centered)

        resampled = nip.conform(img,
                                out_shape=self.inputs.dimensions,
                                voxel_size=voxel_size,
                                order=self.inputs.order,
                                cval=0.0,
                                orientation=self.inputs.orientation,
                                out_class=None)

        # Make sure the resampled data is still in the correct range (cubic spline can mess it up)
        # And save it as float32 to ensure there is no problem.
        vol = img.get_fdata(dtype='f')
        resampled_vol = resampled.get_fdata(dtype='f')
        resampled_vol[resampled_vol < vol.min()] = vol.min()
        resampled_vol[resampled_vol > vol.max()] = vol.max()
        resampled_correct = nib.Nifti1Image(resampled_vol, resampled.affine)

        # Save it for later use in _list_outputs
        _, base, _ = split_filename(fname)
        nib.save(resampled_correct, base + '_resampled.nii.gz')

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.img
        _, base, _ = split_filename(fname)
        outputs["resampled"] = os.path.abspath(base + '_resampled.nii.gz')
        outputs["ori_size"] = self.ori_size
        outputs["ori_resol"] = self.ori_resol
        outputs["ori_orient"] = self.ori_orient
        if self.corrected_affine is not None:
            outputs['corrected_affine'] = self.corrected_affine
        return outputs


class Resample_from_to_InputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply Resample_from_to function on the
    image """
    moving_image = traits.File(exists=True,
                               desc='NIfTI file to resample',
                               mandatory=True)

    fixed_image = traits.File(exists=True,
                              desc='NIfTI file to use as reference for the resampling',
                              mandatory=True)

    spline_order = traits.Int(3,
                              desc="Order of spline interpolation",
                              usedefault=True)

    out_name = traits.Str('resampled.nii.gz',
                          usedefault=True,
                          desc='Output filename')

    out_suffix = traits.Str(mandatory=False,
                            desc=('If set, uses the moving image name and add the given suffix to create the '
                                  'output filename. This will override the "out_name" input of the node.'))

    corrected_affine = traits.Any(desc=('Affine matrix to use instead of the input affine, if defined, '
                                        'as it means that the original space had a bad affine (e.g. img1 '
                                        'needed a correction before its conformation)'))


class Resample_from_to_OutputSpec(TraitedSpec):
    """Output class

    Args:
        conform (nib.Nifti1Image): transformed image
    """
    resampled_image = traits.File(exists=True,
                                  desc='Nifti file of the image after resampling')


class Resample_from_to(BaseInterface):
    """Apply the nibabel function resample_from_to: put a Nifti image in the
    space (dimensions and resolution) of another Nifti image, while keeping
    the correct orientation and spatial position

    Attributes:
        input_spec:
            moving_image: NIfTI file to resample
            fixed_image: NIfTI file used as template for the resampling
            splin_order: order used for the splin interpolation (see scipy.ndimage.affine_transform)
        output_spec:
            resampled_image: Nifti file resampled

    Methods:
        _run_interface(runtime):
            Resample an image based on the dimensions and resolution of another
    """
    input_spec = Resample_from_to_InputSpec
    output_spec = Resample_from_to_OutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Return: runtime
        """
        in_img = nib.load(self.inputs.moving_image)
        if isdefined(self.inputs.corrected_affine):
            # Apply the affine correction to the image before resampling
            # (assuming that in_img is in the same space as the input to the conform node)
            trans_centered = self.inputs.corrected_affine[:3, 3]
            simpl_rot_sign = np.sign(np.diag(self.inputs.corrected_affine[:3, :3]))
            simplified_rot = np.eye(3) * in_img.header['pixdim'][1:4] * simpl_rot_sign  # Keeping the voxel dimensions and signs
            simplified_affine_centered = nib.affines.from_matvec(simplified_rot, trans_centered)
            in_img.set_sform(affine=simplified_affine_centered)
            in_img.set_qform(affine=simplified_affine_centered)
        in_img = nib.funcs.squeeze_image(in_img)
        ref_img = nib.funcs.squeeze_image(nib.load(self.inputs.fixed_image))
        if isdefined(self.inputs.out_suffix):
            basename = os.path.basename(self.inputs.moving_image).split('.nii')[0]
            self.outname = basename + self.inputs.out_suffix + '.nii.gz'
        else:
            self.outname = self.inputs.out_name
        resampled = nip.resample_from_to(in_img,
                                         ref_img,
                                         self.inputs.spline_order)

        nib.save(resampled, self.outname)
        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().trait_get()
        outputs['resampled_image'] = os.path.abspath(self.outname)
        return outputs


class IntensityNormalizationInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply normalization to the nifti
    image"""
    input_image = traits.File(exists=True, desc='NIfTI image input.',
                              mandatory=True)

    percentile = traits.Int(desc='value to threshold above this percentile',
                            mandatory=True)

    brain_mask = traits.File(desc='brain_mask to adapt normalization to '
                             'the greatest number', mandatory=False)

    inverse = traits.Bool(False,
                          desc='If set to True, the normalized value of the voxels in '
                          'the brain will be "inversed" (1-val)',
                          mandatory=False,
                          usedefault=True
                          )


class IntensityNormalizationOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): file img IRM nifti transformed
    """
    intensity_normalized = traits.File(exists=True,
                                       desc='Intensity normalized image')

    mode = traits.Float(desc='Most frequent value of intensities voxel histogram in an interval given')


class Normalization(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image): NIfTI image input
        output_spec (nib.Nifti1Image): Intensity-normalized image

    Methods:
        _run_interface(runtime):
            transformed an image into another with specified arguments
    """
    input_spec = IntensityNormalizationInputSpec
    output_spec = IntensityNormalizationOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # conform image to desired voxel sizes and dimensions
        fname = self.inputs.input_image
        img = nib.load(fname)
        if isdefined(self.inputs.brain_mask) and self.inputs.brain_mask:
            brain_mask = nib.load(self.inputs.brain_mask)
        else:
            brain_mask = None
        img_normalized, mode = normalization(img,
                                             self.inputs.percentile,
                                             brain_mask,
                                             self.inputs.inverse)

        # Save it for later use in _list_outputs
        setattr(self, 'mode', mode)

        _, base, _ = split_filename(fname)
        if self.inputs.inverse:
            self.outname = base + '_img_normalized_inv.nii.gz'
        else:
            self.outname = base + '_img_normalized.nii.gz'
        nib.save(img_normalized, self.outname)

        return runtime

    def _list_outputs(self):
        """Just get the absolute path to the scheme file name."""
        outputs = self.output_spec().get()
        fname = self.inputs.input_image
        _, base, _ = split_filename(fname)
        outputs['mode'] = getattr(self, 'mode')
        outputs["intensity_normalized"] = os.path.abspath(self.outname)
        return outputs


class ThresholdInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply an treshold function on
    nifti image"""
    img = traits.File(exists=True, desc='file img Nifti', mandatory=True)

    threshold = traits.Float(0.5, exists=True, mandatory=True,
                             desc='Value of the treshold to apply to the image'
                             )

    sign = traits.Enum('+', '-',
                       usedefault=True,
                       desc='Whether to keep data above threshold or below threshold.')

    binarize = traits.Bool(False, exists=True,
                           desc='Binarize image')

    open_iter = traits.Int(0, usedefault=True,
                           desc=('For binary opening of the clusters, radius of the ball used '
                                 'as footprint (skip opening if <= 0'))

    clusterCheck = traits.Str('size', usedefault=True,
                              desc=("Can be 'size', 'top' or 'all'. Select one cluster "
                                    "(if not 'all') in the mask, biggest or highest on z-axis"))

    minVol = traits.Int(0, usedefault=True,
                        desc='Minimum size of the clusters to keep (if > 0, else keep all)')

    outname = traits.Str(mandatory=False,
                         desc='name of the output file. If not specified, will be the input witn "_thresholded" appended.')


class ThresholdOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): file img IRM nifti transformed
    """
    thresholded = traits.File(exists=True,
                              desc='Thresholded image')


class Threshold(BaseInterface):
    """Main class

    Attributes:
        input_spec (nib.Nifti1Image):
            file img Nifti
            Threshold for the brain mask
        output_spec (nib.Nifti1Image): file img brain mask IRM nifti

    Methods:
        _run_interface(runtime):
            transformed an image into brain mask
    """
    input_spec = ThresholdInputSpec
    output_spec = ThresholdOutputSpec

    def _run_interface(self, runtime):
        """Run main programm

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        fname = self.inputs.img
        img = nib.funcs.squeeze_image(nib.load(fname))
        thresholded = threshold(img,
                                self.inputs.threshold,
                                sign=self.inputs.sign,
                                binarize=self.inputs.binarize,
                                open_iter=self.inputs.open_iter,
                                clusterCheck=self.inputs.clusterCheck,
                                minVol=self.inputs.minVol)

        # Save it for later use in _list_outputs
        if not isdefined(self.inputs.outname):
            _, base, _ = split_filename(fname)
            outname = base + '_thresholded.nii.gz'
        else:
            outname = self.inputs.outname
        nib.save(thresholded, outname)
        setattr(self, 'outname', os.path.abspath(outname))

        return runtime

    def _list_outputs(self):
        """
        Just gets the absolute path to the scheme file name
        """
        outputs = self.output_spec().get()
        outputs["thresholded"] = getattr(self, 'outname')
        return outputs


class CropInputSpec(BaseInterfaceInputSpec):
    """Input parameter to apply cropping to the
    nifti image"""
    roi_mask = traits.File(exists=True,
                           desc='Mask for computation of center of gravity and'
                           'cropping coordinates', mandatory=False)

    apply_to = traits.File(exists=True,
                           desc='Image to crop', mandatory=True)

    final_dimensions = traits.Tuple(traits.Int, traits.Int, traits.Int,
                                    default=(160, 214, 176),
                                    usedefault=True,
                                    desc='Final image array size in i, j, k.')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc='center of gravity of nifti image cropped with first'
                           'voxel intensities normalization', mandatory=False)

    default = traits.Enum("ijk", "xyz", usedefault=True, desc="Default crop center strategy (voxels or world).")


class CropOutputSpec(TraitedSpec):
    """Output class

    Args:
        img_crop (nib.Nifti1Image): Cropped image
    """
    cropped = traits.File(exists=True,
                          desc='nib.Nifti1Image: preprocessed image')

    cdg_ijk = traits.Tuple(traits.Int, traits.Int, traits.Int,
                           desc="brain_mask's center of gravity")

    bbox1 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box first point')

    bbox2 = traits.Tuple(traits.Int, traits.Int, traits.Int,
                         desc='bounding box second point')

    cdg_ijk_file = traits.File(desc='Saved center of gravity of the brain_mask')

    bbox1_file = traits.File(desc='Saved bounding box first point')

    bbox2_file = traits.File(desc='Saved bounding box second point')


class Crop(BaseInterface):
    """Transform an image to desired dimensions."""
    input_spec = CropInputSpec
    output_spec = CropOutputSpec

    def _run_interface(self, runtime):
        """Run crop function

        Args:
            runtime (_type_): time to execute the
            function
        Return: runtime
        """
        # load images
        if isdefined(self.inputs.roi_mask):
            mask = nib.load(self.inputs.roi_mask)
        else:
            # get from ijk
            mask = None
        if not isdefined(self.inputs.cdg_ijk):
            cdg_ijk = None
        else:
            cdg_ijk = self.inputs.cdg_ijk
        target = nib.load(self.inputs.apply_to)

        # process
        cropped, cdg_ijk, bbox1, bbox2 = crop(
            mask,
            target,
            self.inputs.final_dimensions,
            cdg_ijk,
            self.inputs.default,
            safety_marger=5)

        # TODO: rewrite this using np.savetxt and change reader where needed
        cdg_ijk = cdg_ijk[0], cdg_ijk[1], cdg_ijk[2]
        bbox1 = bbox1[0], bbox1[1], bbox1[2]
        bbox2 = bbox2[0], bbox2[1], bbox2[2]

        with open('cdg_ijk.txt', 'w') as fid:
            fid.write(str(cdg_ijk))
        cdg_ijk_file = os.path.abspath('cdg_ijk.txt')
        with open('bbox1.txt', 'w') as fid:
            fid.write(str(bbox1))
        bbox1_file = os.path.abspath('bbox1.txt')
        with open('bbox2.txt', 'w') as fid:
            fid.write(str(bbox2))
        bbox2_file = os.path.abspath('bbox2.txt')

        # Save it for later use in _list_outputs
        setattr(self, 'cdg_ijk', cdg_ijk)
        setattr(self, 'bbox1', bbox1)
        setattr(self, 'bbox2', bbox2)
        setattr(self, 'cdg_ijk_file', cdg_ijk_file)
        setattr(self, 'bbox1_file', bbox1_file)
        setattr(self, 'bbox2_file', bbox2_file)

        _, base, _ = split_filename(self.inputs.apply_to)
        nib.save(cropped, base + '_cropped.nii.gz')

    def _list_outputs(self):
        """Fill in the output structure."""
        outputs = self.output_spec().trait_get()
        fname = self.inputs.apply_to
        outputs['cdg_ijk'] = getattr(self, 'cdg_ijk')
        outputs['bbox1'] = getattr(self, 'bbox1')
        outputs['bbox2'] = getattr(self, 'bbox2')
        outputs['cdg_ijk_file'] = getattr(self, 'cdg_ijk_file')
        outputs['bbox1_file'] = getattr(self, 'bbox1_file')
        outputs['bbox2_file'] = getattr(self, 'bbox2_file')
        _, base, _ = split_filename(fname)
        outputs["cropped"] = os.path.abspath(base + '_cropped.nii.gz')

        return outputs


class PredictInputSpec(BaseInterfaceInputSpec):
    """Predict input specification."""
    models = traits.List(traits.File(exists=True),
                         argstr='-m %s',
                         desc='Model files.',
                         mandatory=False,
                         )

    img = traits.File(argstr='--img %s',
                      desc='The T1W image of the subject.',
                      mandatory=False,
                      exists=True)

    model = traits.Directory('/mnt/model',
                             argstr='--model %s',
                             exists=False,
                             desc='Folder containing hdf5 model files.',
                             usedefault=True
                             )

    descriptor = traits.File(argstr='-descriptor %s',
                             exists=True,
                             desc='File information about models for validation',
                             mandatory=True)

    gpu_number = traits.Int(argstr='--gpu %d',
                            desc='GPU to use if several GPUs are available.',
                            mandatory=False)

    verbose = traits.Bool(True,
                          argstr='--verbose',
                          desc='Verbose output',
                          mandatory=False)

    threads = traits.Int(argstr='--threads %d',
                         desc='Number of threads to use when running on CPU.',
                         mandatory=False)

    out_filename = traits.Str('map.nii.gz',
                              argstr='-o %s',
                              desc='Output filename.',
                              usedefault=True)


class PredictOutputSpec(TraitedSpec):
    segmentation = traits.File(desc='The segmentation image',
                               exists=True)


class Predict(CommandLine):
    """Run predict to segment from reformated structural images.

    Uses a 3D U-Net.
    """
    input_spec = PredictInputSpec
    output_spec = PredictOutputSpec
    _cmd = 'defacer_predict'  # defacer.predict:main

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["segmentation"] = os.path.abspath(os.path.basename(str(self.inputs.out_filename)))
        return outputs


class Nii2dcmInputSpec(BaseInterfaceInputSpec):
    """nii2dcm input specification."""
    nii_input = traits.File(position=0,
                            argstr="%s",
                            desc='Nifti file to convert to DICOM',
                            mandatory=True,
                            exists=True,
                            )
    out_dir = traits.Directory(position=1,
                               argstr="%s",
                               desc='Output folder for the new DICOM series',
                               mandatory=True,
                               )
    dcm_type = traits.Enum('MR', 'SVR',
                           argstr="--dicom_type %s",
                           desc=("DICOM type. MR folr multi-slice 2D MRI, SVR for 3D SVR (slice-to-volume registration) MRI. "
                                 "If not specified, will create a generic DICOM output."))

    dcm_ref = traits.File(argstr="--ref_dicom %s",
                          desc='DICOM file or folder to use to transfer metadata to the new DICOM series')


class Nii2dcmOutputSpec(TraitedSpec):
    out_dir = traits.Directory(desc='Output folder for the new DICOM series',
                               exists=True)


class Nii2dcm(CommandLine):
    """Run nii2dcm

    """
    input_spec = Nii2dcmInputSpec
    output_spec = Nii2dcmOutputSpec
    _cmd = 'nii2dcm'  # defacer.predict:main

    def _run_interface(self, runtime, correct_return_codes=(0,)):
        out_dir = Path(self.inputs.out_dir)
        if not out_dir.exists():
            out_dir.mkdir()
            self.inputs.out_dir = str(out_dir)
        dcm_ref = Path(self.inputs.dcm_ref)
        if dcm_ref.is_file():
            pass
        elif dcm_ref.is_dir():
            isdcm = False
            for infile in dcm_ref.iterdir():
                if infile.is_file():
                    with open(infile, "rb") as fp:
                        fp.read(128)
                        isdcm = (fp.read(4) == b"DICM")
                    if isdcm:
                        break
            if isdcm:
                self.inputs.dcm_ref = str(infile)
            else:
                raise ValueError('No DICOM file found for the dcm_ref input of the Nii2dcm interface')
        return super()._run_interface(runtime, correct_return_codes)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_dir"] = Path(self.inputs.out_dir).absolute()
        return outputs
