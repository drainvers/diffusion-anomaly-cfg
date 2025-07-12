import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
from scipy import ndimage
from PIL import Image
from textwrap import wrap
import random

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class VinDRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory,
        class_cond=True,
        test_flag=False,
        sample_n=None,
        transform=None,
    ):
        if not root_directory:
            raise ValueError("unspecified data directory")
        self.directory = os.path.expanduser(root_directory)
        self.test_flag = test_flag
        self.class_cond = class_cond
        self.annotations_path = os.path.join(self.directory,
                                             "annotations/image_labels_test.csv"
                                             if self.test_flag else
                                             "annotations/image_labels_train.csv")
        self.transform = transform
        df_annotations = pd.read_csv(self.annotations_path)

        if not test_flag:
            df_annotations = self._clean_df(df_annotations)

        self.local_classes = None
        if class_cond:
            df_annotations.loc[(df_annotations["No finding"] == 1), 'Label'] = 0 # Healthy
            df_annotations.loc[(df_annotations["Pleural effusion"] == 1), 'Label'] = 1 # Diseased

            df_annotations_healthy = df_annotations[(df_annotations["Label"] == 0)]
            df_annotations_diseased = df_annotations[(df_annotations["Label"] == 1)]

            # Sample from pleural effusion images to lower imbalance
            if sample_n:
                if len(df_annotations_healthy.index) > sample_n:
                    df_annotations_healthy = df_annotations_healthy.sample(n=sample_n, random_state=1911)
                if len(df_annotations_diseased.index) > sample_n:
                    df_annotations_diseased = df_annotations_diseased.sample(n=sample_n, random_state=1911)

            df_annotations = pd.concat([df_annotations_diseased, df_annotations_healthy]).copy(deep=True)

            self.local_classes = df_annotations['Label'].to_list()
        
        df_annotations['Path'] = df_annotations['image_id'].apply(
                                    lambda image_id: os.path.join(
                                        self.directory, 
                                        f"{'test' if test_flag else 'train'}/{image_id}.jpg")).astype(str)
        self.local_images = df_annotations['Path'].to_list()

        if class_cond:
            assert len(self.local_images) == len(self.local_classes)


    def __getitem__(self, idx):
        path = self.local_images[idx]
        basename, ext = os.path.splitext(os.path.basename(path))
        name = basename
        print(name)

        # Readds ability to read known image formats, taken from upstream guided diffusion repo
        if ext == '.npy':
            out_img = np.load(path)
        else:
            out_img = np.asarray(Image.open(path).convert('L')) # Use Pillow for TIF support
            out_img = np.expand_dims(out_img, axis=2) # Changes image shape to (H, W, 1)
        
        out_img = visualize(out_img).astype(np.float32)
        out_img = np.transpose(out_img, [2, 0, 1]) # HWC -> CHW

        out_dict = {}
        if self.local_classes:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["name"] = name
        
        if self.transform:
            out_img = self.transform(out_img)

        return out_img, out_dict
    
    def summarize(self):
        print('n_images  :', len(self.local_images))
        if self.class_cond:
            print('n_labels  :', len(self.local_classes))
            print('n_healthy :', len(self.local_classes) - int(sum(self.local_classes)))
            print('n_diseased:', int(sum(self.local_classes)))
    
    def _clean_df(self, input_df):
        pathology_columns = [
            "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", 
            "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA", 
            "ILD", "Infiltration", "Lung Opacity", "Lung cavity", "Lung cyst", 
            "Mediastinal shift", "Nodule/Mass", "Pleural effusion", "Pleural thickening", 
            "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Other lesion", "COPD", 
            "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", "No finding"
        ]

        # Function to apply majority voting per group
        def majority_vote(group):
            # Use sum > 1 to enforce 2/3 votes for positive
            majority = (group[pathology_columns].sum(axis=0) > 1).astype(int)
            # If 'No finding' is 1, set all others to 0
            if majority["No finding"] == 1:
                for col in pathology_columns:
                    if col != "No finding":
                        majority[col] = 0
            # Return as a DataFrame row with the image_id
            majority['image_id'] = group['image_id'].iloc[0]
            return majority

        # Apply majority voting grouped by image_id
        result_df = input_df.groupby("image_id").apply(majority_vote).reset_index(drop=True)

        return result_df

    def __len__(self):
        return len(self.local_images)

class ChexpertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_directory,
        class_cond=True,
        test_flag=False,
        sample_n=None,
        transform=None,
        data_filter=None,
        frontal_mode="both",
        unique_patients_only=False
    ):
        if not root_directory:
            raise ValueError("unspecified data directory")
        self.directory = os.path.expanduser(root_directory)
        self.test_flag = test_flag
        self.class_cond = class_cond
        self.annotations_path = os.path.join(self.directory,
                                             "valid.csv"
                                             if self.test_flag else
                                             "train.csv")
        self.transform = transform

        # Filtering logic
        df_annotations = pd.read_csv(self.annotations_path)

        if data_filter == "frontal_only":
            df_annotations = df_annotations[(df_annotations["Frontal/Lateral"] == "Frontal")].copy(deep=True)
            if frontal_mode == "ap":
                df_annotations = df_annotations[(df_annotations["AP/PA"] == "AP")].copy(deep=True)
            elif frontal_mode == "pa":
                df_annotations = df_annotations[(df_annotations["AP/PA"] == "PA")].copy(deep=True)

        self.local_classes = None
        if class_cond:
            # Generate labels for healthy images and images with pleural effusions
            df_annotations.loc[(df_annotations["No Finding"] == 1), 'Label'] = 0 # Healthy
            df_annotations.loc[(df_annotations["Pleural Effusion"] == 1), 'Label'] = 1 # Diseased
            df_annotations = df_annotations[df_annotations['Label'].isin([0, 1])].copy(deep=True)

            df_annotations_healthy = df_annotations[(df_annotations["Label"] == 0)]
            df_annotations_diseased = df_annotations[(df_annotations["Label"] == 1)]

            if unique_patients_only:
                df_annotations_healthy = self._clean_df(df_annotations_healthy)
                df_annotations_diseased = self._clean_df(df_annotations_diseased)

            # Sample from pleural effusion images to lower imbalance
            if sample_n:
                if len(df_annotations_healthy.index) > sample_n:
                    df_annotations_healthy = df_annotations_healthy.sample(n=sample_n, random_state=1911)
                if len(df_annotations_diseased.index) > sample_n:
                    df_annotations_diseased = df_annotations_diseased.sample(n=sample_n, random_state=1911)

            df_annotations = pd.concat([df_annotations_diseased, df_annotations_healthy]).copy(deep=True)

            self.local_classes = df_annotations['Label'].to_list()
        
        df_annotations['Path'] = df_annotations['Path'].apply(
                                    lambda path: os.path.join(
                                        self.directory, 
                                        os.sep.join(path.split(os.sep)[1:]))).astype(str)
        self.local_images = df_annotations['Path'].to_list()

        if class_cond:
            assert len(self.local_images) == len(self.local_classes)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        basename, ext = os.path.splitext(os.path.basename(path))
        name = basename
        print(name)

        # Readds ability to read known image formats, taken from upstream guided diffusion repo
        if ext == '.npy':
            out_img = np.load(path)
        else:
            out_img = np.asarray(Image.open(path).convert('L')) # Use Pillow for TIF support
            out_img = np.expand_dims(out_img, axis=2) # Changes image shape to (H, W, 1)
        
        out_img = visualize(out_img).astype(np.float32)
        out_img = np.transpose(out_img, [2, 0, 1]) # HWC -> CHW

        out_dict = {}
        if self.local_classes:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            out_dict["name"] = name
        
        if self.transform:
            out_img = self.transform(out_img)

        return out_img, out_dict
    
    def summarize(self):
        print('n_images  :', len(self.local_images))
        if self.class_cond:
            print('n_labels  :', len(self.local_classes))
            print('n_healthy :', len(self.local_classes) - int(sum(self.local_classes)))
            print('n_diseased:', int(sum(self.local_classes)))
    
    def _clean_df(self, input_df):
        # Do NOT use first(), it will return the first non-NaN value, use nth to get values as-is
        input_df.loc[:, 'Patient ID'] = input_df['Path'].map(lambda x: os.path.basename(x).split('_')[0])
        return input_df.groupby('Patient ID').nth(0).reset_index(drop=True)

    def __len__(self):
        return len(self.local_images)

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number=filedict['t1'].split('/')[4]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(
                number) + '-label.nii.gz'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database)

def preview_dataset(ds, nrow=6, ncol=5, save_path="preview.png"):
    # Create the figure
    fig = plt.figure(figsize=(ncol * 2.56, nrow * 2.56), dpi=100, layout='tight')
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.2, hspace=0.2)  # Spacing between images

    for grid_idx in range(nrow * ncol):
        row, col = divmod(grid_idx, ncol)
        try:
            im, _ = ds[grid_idx]
        except Exception:
            im = np.ones((3, 256, 256), dtype=np.uint8) * 255

        # Convert tensor to numpy and reshape if necessary
        if torch.is_tensor(im):
            im = im.detach().cpu().numpy()
        if im.shape[0] in [1, 3]:  # C x H x W format
            im = np.transpose(im, (1, 2, 0))  # H x W x C

        # Clip and normalize if needed
        im = np.clip(im, 0, 1) if im.dtype != np.uint8 else im

        ax = plt.subplot(gs[row, col])
        ax.imshow(im, cmap='gray' if im.shape[-1] == 1 else None, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # train_pa = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=False, data_filter="frontal_only", frontal_mode="pa")
    # train_ap = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=False, data_filter="frontal_only", frontal_mode="ap")
    # train_ds = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=False, data_filter="frontal_only")
    # test_ds = ChexpertDataset('/workspace/CheXpert-v1.0', class_cond=True, test_flag=True, data_filter="frontal_only")
    train_ds = VinDRDataset('/workspace/vindr_cxr_256', class_cond=False, test_flag=False)
    # test_ds = VinDRDataset('/workspace/vindr_cxr_256', class_cond=True, test_flag=True)
    preview_dataset(train_ds, 5, 6, 'vindr_preview.png')

    # import blobfile as bf
    # from torch.utils.data import DataLoader, Dataset

    # def visualize(img):
    #     _min = img.min()
    #     _max = img.max()
    #     normalized_img = (img - _min)/ (_max - _min)
    #     return normalized_img

    # class ImageDataset(Dataset):
    #     def __init__(
    #         self,
    #         resolution,
    #         image_paths,
    #         classes=None,
    #         shard=0,
    #         num_shards=1,
    #         random_crop=False,
    #         random_flip=False
    #     ):
    #         super().__init__()
    #         self.resolution = resolution
    #         self.local_images = image_paths[shard:][::num_shards]
    #         self.local_classes = None if classes is None else classes[shard:][::num_shards]
    #         self.random_crop = random_crop
    #         self.random_flip = random_flip

    #     def __len__(self):
    #         print('len',  len(self.local_images))
    #         return len(self.local_images)

    #     def __getitem__(self, idx):
    #         path = self.local_images[idx]
    #         # name=str(path).split("/")[-1].split(".")[0]
    #         name, ext = os.path.splitext(os.path.basename(path))
    #         print('name', name)

    #         # Readds ability to read known image formats, taken from upstream guided diffusion repo
    #         if ext == '.npy':
    #             numpy_img = np.load(path)
    #         else:
    #             numpy_img = np.asarray(Image.open(path).convert('L')) # Use Pillow for TIF support
    #             # numpy_img = cv2.resize(numpy_img, (256, 256), interpolation=cv2.INTER_AREA)
    #             numpy_img = np.expand_dims(numpy_img, axis=2) # Changes image shape to (W, H, 1)
    #         arr = visualize(numpy_img).astype(np.float32)

    #         out_dict = {}
    #         if self.local_classes is not None:
    #             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
    #             out_dict["name"]=name

    #         return np.transpose(arr, [2, 0, 1]), out_dict # HWC -> CHW

    # def _list_image_files_recursively(data_dir):
    #     results = []
    #     for entry in sorted(bf.listdir(data_dir)):
    #         full_path = bf.join(data_dir, entry)
    #         ext = os.path.splitext(os.path.basename(full_path))[1]
    #         if ext.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".npy"]:
    #             results.append(full_path)
    #         elif bf.isdir(full_path):
    #             results.extend(_list_image_files_recursively(full_path))
    #     results = sorted(results)
    #     return results
    
    # all_files = _list_image_files_recursively('../data/tif_pleural/')
    # class_names =[path.split("/")[-2] for path in all_files] #9 or 3
    # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names), reverse=True))}
    # if len(sorted_classes) == 1 and sorted_classes.get('diseased') == 0:
    #         sorted_classes['diseased'] = 1
    # print('sorted_classes', sorted_classes)
    # classes = [sorted_classes[x] for x in class_names]

    # dataset = ImageDataset(
    #     256,
    #     all_files,
    #     classes=classes,
    #     shard=0,
    #     num_shards=1,
    #     random_crop=False,
    #     random_flip=False,
    # )
    # preview_dataset(dataset, 3, 4, 'tif_pleural_preview.png')