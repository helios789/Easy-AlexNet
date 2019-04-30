import numpy as np
import os
import cv2

class DataSet():

    def __init__(self, subset_dir):
        self.images, self.labels = self.read_data(subset_dir)
    

    def read_data(self, subset_dir, one_hot=True):
        # 학습+검증 데이터셋을 읽어들임
        filename_list = os.listdir(subset_dir)
        set_size = len(filename_list)

        # 단순히 filename list의 순서를 랜덤하게 섞음
        np.random.shuffle(filename_list)

        # 데이터 array들을 메모리 공간에 미리 할당함
        X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
        y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)

        for i, filename in enumerate(filename_list):
            
            # labeling
            label = filename.split('.')[0]
            if label == 'cat':
                y = 0
            else:  # label == 'dog'
                y = 1

            # load images
            file_path = os.path.join(subset_dir, filename)
            img = cv2.imread(file_path)    # shape: (H, W, 3), range: [0, 255]
            img = cv2.resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]

            # save images and labels
            X_set[i] = img
            y_set[i] = y

            if i % 1000 == 0:
                print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')

        if one_hot:
            # 모든 레이블들을 one-hot 인코딩 벡터들로 변환함, shape: (N, num_classes)
            y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
            y_set_oh[np.arange(set_size), y_set] = 1
            y_set = y_set_oh

        return X_set, y_set


    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True, fake_data=False):
        """
        `batch_size` 개수만큼의 이미지들을 현재 데이터셋으로부터 추출하여 미니배치 형태로 반환함.
        :param batch_size: int, 미니배치 크기.
        :param shuffle: bool, 미니배치 추출에 앞서, 현재 데이터셋 내 이미지들의 순서를 랜덤하게 섞을 것인지 여부.
        :param augment: bool, 미니배치를 추출할 때, 데이터 증강을 수행할 것인지 여부.
        :param is_train: bool, 미니배치 추출을 위한 현재 상황(학습/예측).
        :param fake_data: bool, (디버깅 목적으로) 가짜 이미지 데이터를 생성할 것인지 여부.
        :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if fake_data:
            fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
            fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
            fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
            return fake_batch_images, fake_batch_labels

        start_index = self._index_in_epoch

        # 맨 첫 번째 epoch에서는 전체 데이터셋을 랜덤하게 섞음
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # 현재의 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch을 진행함
        if start_index + batch_size > self._num_examples:
            # 완료된 epochs 수를 1 증가
            self._epochs_completed += 1
            # 새로운 epoch에서, 남은 이미지들을 가져옴
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # 하나의 epoch이 끝나면, 전체 데이터셋을 섞음
            if shuffle:
                np.random.shuffle(self._indices)

            # 다음 epoch 시작
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # 학습 상황에서의 데이터 증강을 수행함 
            batch_images = random_crop_reflect(batch_images, 227)
        elif augment and not is_train:
            # 예측 상황에서의 데이터 증강을 수행함
            batch_images = corner_center_crop_reflect(batch_images, 227)
        else:
            # 데이터 증강을 수행하지 않고, 단순히 이미지 중심 위치에서만 추출된 패치를 사용함
            batch_images = center_crop(batch_images, 227)

        return batch_images, batch_labels