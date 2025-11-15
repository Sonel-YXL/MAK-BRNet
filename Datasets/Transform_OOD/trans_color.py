import numpy as np
import cv2
import time
import random
import os
from skimage.util import random_noise
from skimage import exposure

class to_RGB():
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        if len(data['img'].shape) == 2:
            img = np.expand_dims(data['img'], axis=2)
            data['img'] = img.repeat(3,axis=2)
        return data


class to_grayscaled():
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        h, w, c = data['img'].shape
        if c == 3:
            img = np.expand_dims(
                0.2125 * data['img'][:, :, 0] + 0.7154 * data['img'][:, :, 1] + 0.0721 * data['img'][:, :, 2],
                axis=2).astype(np.uint8)
            data['img'] = img
        return data

class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate


    def __call__(self, image, annotation):
        """
        对图片加噪声
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return image, annotation
        res_img = (random_noise(image, mode='gaussian', clip=True) * 255).astype(image.dtype)
        return res_img, annotation






class ChangeLight:
    def __init__(self, random_rate):
        self.random_rate = random_rate
        pass


    def __call__(self, image, annotation):
        if random.random() > self.random_rate:
            return image, annotation
        img = image
        flag = random.uniform(0.5, 1.5)

        label = round(flag, 2)


        img_gamma = exposure.adjust_gamma(img, flag)

        return img_gamma, annotation


class AddGaussianNoise:
    def __init__(self, mean=0, var=0.01):
        self.mean = mean
        self.var = var

    def __call__(self, data: dict) -> dict:
        img = data['img']
        img = np.array(img / 255, dtype=float)

        noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
        out = img + noise

        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)

        data['img'] = out
        return data


class ContrastAdjust:
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        img = data['img']
        img_shape = img.shape

        temp_img = np.zeros(img_shape, dtype=np.float32)
        for num in range(0, 3):

            in_image = img[:, :, num]

            Imax = np.max(in_image)
            Imin = np.min(in_image)

            Omin, Omax = 0, 255

            a = float(Omax - Omin) / (Imax - Imin)
            b = Omin - a * Imin

            out_image = a * in_image + b

            out_image = out_image.astype(np.uint8)
            temp_img[:, :, num] = out_image

        data['img'] = temp_img
        return data


def demo_transform(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    data = {'img': img}


    change_light = ChangeLight(random_rate=0.99)
    data = change_light(data)
    print(f"Light adjustment label: {data.get('label', None)}")














    processed_img = data['img']


    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)


    cv2.imshow("Original Image", img)
    cv2.imshow("Processed Image", processed_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import cv2
    import numpy as np




    demo_transform("./demo.png")