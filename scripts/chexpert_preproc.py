import pandas as pd
import numpy as np
import os
import cv2
import tqdm

tqdm.tqdm.pandas()

class ChexpertResNormPipeline:
    def __init__(self, root_path):
        self.root_path = root_path
        self.df_train = pd.read_csv(f'{self.root_path}/CheXpert-v1.0/train.csv')
        self.df_valid = pd.read_csv(f'{self.root_path}/CheXpert-v1.0/valid.csv')

        self.image_paths = np.concatenate((self.df_train['Path'].to_numpy(), self.df_valid['Path'].to_numpy()))

        print(f"Train rows: {self.df_train.shape[0]}")
        print(f"Valid rows: {self.df_valid.shape[0]}")

    def _preprocess_image(self, path, target_size, normalization_type='histeq'):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        width, height = target_size, target_size

        if normalization_type == 'histeq':
            resized = cv2.equalizeHist(img)
        else:
            resized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        resized = cv2.resize(resized, (width, height), interpolation=cv2.INTER_AREA)
        resized = np.expand_dims(resized, axis=2) # Model input must be (H, W, 1)
    
        return resized

    def _update_label_paths(self, label_df, dest_prefix, split, ext='.npy', chunk_size=10):
        df = label_df.copy(deep=True)
        df.loc[:, 'Path'] = df['Path'].progress_apply(lambda x: self._generate_new_path(x, ext))
        chunks = np.array_split(df.index, chunk_size)
        os.makedirs(f'{dest_prefix}/CheXpert-v1.0/', exist_ok=True)

        for chunk, subset in enumerate(tqdm.tqdm(chunks, desc=f'Saving new {split} CSV with updated paths...')):
            if chunk == 0: # First row
                df.loc[subset].to_csv(f'{dest_prefix}/CheXpert-v1.0/{split}.csv', mode='w', index=False)
            else:
                df.loc[subset].to_csv(f'{dest_prefix}/CheXpert-v1.0/{split}.csv', mode='a', header=None, index=False)
    
    def _generate_new_path(self, old_path, new_ext):
        split_path = old_path.split('/')
        basepath = '/'.join(split_path[:2])
        filename = os.path.splitext('_'.join(split_path[2:]))[0] + new_ext
        return '/'.join([basepath, filename])

    def execute(self, dest_prefix, target_size, ext='.jpg'):
        self._update_label_paths(self.df_train, dest_prefix, 'train', ext)
        self._update_label_paths(self.df_valid, dest_prefix, 'valid', ext)

        for img_path in tqdm.tqdm(self.image_paths, desc=f"Resizing and normalizing images..."):
            new_path = self._generate_new_path(img_path, ext)
            basepath = os.path.dirname(new_path)
            filename = os.path.basename(new_path)

            if not os.path.exists(f'{dest_prefix}/{basepath}'):
                os.makedirs(f'{dest_prefix}/{basepath}', exist_ok=True)

            final_image = self._preprocess_image(f'{self.root_path}/{img_path}', target_size)

            if ext == '.jpg':
                cv2.imwrite(f'{dest_prefix}/{basepath}/{filename}', final_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            elif ext == '.png':
                cv2.imwrite(f'{dest_prefix}/{basepath}/{filename}', final_image)
            else: # Assume .npy
                with open(f'{dest_prefix}/{basepath}/{filename}', 'wb') as img_file:
                    np.save(img_file, final_image)

class ChexpertFilterPipeline:
    def __init__(self, root_path):
        self.root_path = root_path
        self.df_train = pd.read_csv(f'{self.root_path}/CheXpert-v1.0/train.csv')
        self.df_valid = pd.read_csv(f'{self.root_path}/CheXpert-v1.0/valid.csv')

        self.df_train_frontal = self.df_train[(self.df_train['Frontal/Lateral'] == 'Frontal')].copy()
        self.df_train_lateral = self.df_train[(self.df_train['Frontal/Lateral'] == 'Lateral')].copy()
    
        self.df_valid_frontal = self.df_valid[(self.df_valid['Frontal/Lateral'] == 'Frontal')].copy()
        self.df_valid_lateral = self.df_valid[(self.df_valid['Frontal/Lateral'] == 'Lateral')].copy()

        self.df_train_frontal_healthy  = self.df_train_frontal[(self.df_train_frontal['No Finding'] == 1)].copy(deep=True)
        self.df_train_lateral_healthy  = self.df_train_lateral[(self.df_train_lateral['No Finding'] == 1)].copy(deep=True)
        self.df_train_frontal_diseased = self.df_train_frontal[(self.df_train_frontal['Pleural Effusion'] == 1)].copy(deep=True)
        self.df_train_lateral_diseased = self.df_train_lateral[(self.df_train_lateral['Pleural Effusion'] == 1)].copy(deep=True)

        self.df_valid_frontal_healthy  = self.df_valid_frontal[(self.df_valid_frontal['No Finding'] == 1)].copy(deep=True)
        self.df_valid_lateral_healthy  = self.df_valid_lateral[(self.df_valid_lateral['No Finding'] == 1)].copy(deep=True)
        self.df_valid_frontal_diseased = self.df_valid_frontal[(self.df_valid_frontal['Pleural Effusion'] == 1)].copy(deep=True)
        self.df_valid_lateral_diseased = self.df_valid_lateral[(self.df_valid_lateral['Pleural Effusion'] == 1)].copy(deep=True)

        self.df_train_frontal_healthy_paths  = self.df_train_frontal_healthy['Path'].to_numpy()
        self.df_train_lateral_healthy_paths  = self.df_train_lateral_healthy['Path'].to_numpy()
        self.df_train_frontal_diseased_paths = self.df_train_frontal_diseased['Path'].to_numpy()
        self.df_train_lateral_diseased_paths = self.df_train_lateral_diseased['Path'].to_numpy()

        self.df_valid_frontal_healthy_paths  = self.df_valid_frontal_healthy['Path'].to_numpy()
        self.df_valid_lateral_healthy_paths  = self.df_valid_lateral_healthy['Path'].to_numpy()
        self.df_valid_frontal_diseased_paths = self.df_valid_frontal_diseased['Path'].to_numpy()
        self.df_valid_lateral_diseased_paths = self.df_valid_lateral_diseased['Path'].to_numpy()

        print(f"Train rows: {self.df_train.shape[0]}")
        print(f"Valid rows: {self.df_valid.shape[0]}")

        print(f"Frontal healthy train rows : {self.df_train_frontal_healthy.shape[0]}")
        print(f"Frontal diseased train rows: {self.df_train_frontal_diseased.shape[0]}")

        print(f"Lateral healthy train rows : {self.df_train_lateral_healthy.shape[0]}")
        print(f"Lateral diseased train rows: {self.df_train_lateral_diseased.shape[0]}")

        print(f"Frontal healthy valid rows : {self.df_valid_frontal_healthy.shape[0]}")
        print(f"Frontal diseased valid rows: {self.df_valid_frontal_diseased.shape[0]}")

        print(f"Lateral healthy valid rows : {self.df_valid_lateral_healthy.shape[0]}")
        print(f"Lateral diseased valid rows: {self.df_valid_lateral_diseased.shape[0]}")

    def _get_patient_name(self, path):
        filename = os.path.basename(path)
        patient_name = filename.split('_')[0]
        return patient_name
    
    def _get_filename_from_path(self, path):
        path_elems = path.split('/')
        return os.path.splitext(path_elems[-1])[0]
    
    def _clean_df(self, input_df):
        # Do NOT use first(), it will return the first non-NaN value, use nth to get values as-is
        input_df.loc[:, 'Patient ID'] = input_df['Path'].map(self._get_patient_name)
        return input_df.groupby('Patient ID').nth(0).reset_index(drop=True)
    
    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if os.path.splitext(path)[1] != '.npy':
            img = np.expand_dims(img, axis=2)
        return img

    def _filter_images(self, dest_prefix, paths, split, orientation, class_type, ext='.png'):
        for img_path in tqdm.tqdm(paths, desc=f"Filtering {orientation} {class_type} {split} images..."):
            if os.path.exists(f'{self.root_path}/{img_path}'):
                final_image = self._read_image(f'{self.root_path}/{img_path}')
                filename = self._get_filename_from_path(img_path)

                if ext == '.jpg':
                    cv2.imwrite(f'{dest_prefix}/{split}/{orientation}/{class_type}/{filename}{ext}', final_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                elif ext == '.png':
                    cv2.imwrite(f'{dest_prefix}/{split}/{orientation}/{class_type}/{filename}{ext}', final_image)
                else: # Assume .npy
                    with open(f'{dest_prefix}/{split}/{orientation}/{class_type}/{filename}{ext}', 'wb') as img_file:
                        np.save(img_file, final_image)
    
    def execute(self, dest_prefix):
        folder_paths = [f'{dest_prefix}/training/frontal/healthy',
                        f'{dest_prefix}/training/frontal/diseased',
                        f'{dest_prefix}/training/lateral/healthy',
                        f'{dest_prefix}/training/lateral/diseased',
                        f'{dest_prefix}/testing/frontal/healthy',
                        f'{dest_prefix}/testing/frontal/diseased',
                        f'{dest_prefix}/testing/lateral/healthy',
                        f'{dest_prefix}/testing/lateral/diseased']
        
        for path in folder_paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        self._filter_images(dest_prefix, self.df_train_frontal_healthy_paths,  'training', 'frontal', 'healthy')
        self._filter_images(dest_prefix, self.df_train_frontal_diseased_paths, 'training', 'frontal', 'diseased')
        self._filter_images(dest_prefix, self.df_train_lateral_healthy_paths,  'training', 'lateral', 'healthy')
        self._filter_images(dest_prefix, self.df_train_lateral_diseased_paths, 'training', 'lateral', 'diseased')
        self._filter_images(dest_prefix, self.df_valid_frontal_healthy_paths,  'testing' , 'frontal', 'healthy')
        self._filter_images(dest_prefix, self.df_valid_frontal_diseased_paths, 'testing' , 'frontal', 'diseased')
        self._filter_images(dest_prefix, self.df_valid_lateral_healthy_paths,  'testing' , 'lateral', 'healthy')
        self._filter_images(dest_prefix, self.df_valid_lateral_diseased_paths, 'testing' , 'lateral', 'diseased')

task = str(input('ResNorm/Filter? [R/F]: '))

if task == 'R':
    chexpert_path = str(input('Enter path to CheXpert dataset: '))
    assert(os.path.exists(f"{chexpert_path}/CheXpert-v1.0"))
    dest_path = str(input('Enter destination path: '))
    target_size = int(input('Enter target image size (used for both W and H): '))

    resnorm = ChexpertResNormPipeline(chexpert_path)
    resnorm.execute(dest_path, target_size)
elif task == 'F':
    chexpert_path = str(input('Enter path to CheXpert dataset: '))
    dest_path = str(input('Enter destination path: '))

    filtering = ChexpertFilterPipeline(chexpert_path)
    filtering.execute(dest_path)