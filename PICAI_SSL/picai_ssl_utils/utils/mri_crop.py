import os
import numpy as np
import nibabel as nib

def crop_roi_around_label(mri_image, label_mask, margin=10):
    coords = np.array(np.nonzero(label_mask))
    if coords.size == 0:
        raise ValueError("Label mask is empty!")

    min_coords = np.maximum(np.min(coords, axis=1) - margin, 0)
    max_coords = np.minimum(np.max(coords, axis=1) + margin + 1, label_mask.shape)

    slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    cropped_image = mri_image[slices]
    cropped_mask = label_mask[slices]

    return cropped_image, cropped_mask, slices

def main():
    # 🧠 Hardcoded paths — update as needed
    image_path = r"E:\Dataset\Picai Dataset\picai_public_images_fold0\10000\10000_1000000_t2w.mha"
    label_path = r"E:\Dataset\Picai Dataset\picai_labels-main\anatomical_delineations\whole_gland\AI\Bosma22b\10000_1000000.nii.gz\10000_1000000.nii"
    output_image_path = r"E:\Dataset\Picai Dataset\Croped image"
    output_label_path = r"E:\Dataset\Picai Dataset\Croped image"
    margin = 10

    # Load data
    mri_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    mri_data = mri_nii.get_fdata()
    label_data = label_nii.get_fdata().astype(np.uint8)  # ensure binary

    print(f"Original MRI shape: {mri_data.shape}")
    print(f"Original Label shape: {label_data.shape}")

    # Crop
    cropped_img, cropped_lbl, crop_slices = crop_roi_around_label(mri_data, label_data, margin=margin)

    print(f"Cropped MRI shape: {cropped_img.shape}")
    print(f"Cropped Label shape: {cropped_lbl.shape}")

    # Adjust affine (origin shift)
    affine = mri_nii.affine.copy()
    start = [s.start for s in crop_slices]
    new_origin = nib.affines.apply_affine(affine, start)
    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin

    # Save outputs
    nib.save(nib.Nifti1Image(cropped_img, new_affine), output_image_path)
    nib.save(nib.Nifti1Image(cropped_lbl, new_affine), output_label_path)
    print("✅ Cropped MRI and label saved successfully.")

if __name__ == "__main__":
    main()
