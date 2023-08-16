import os
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import Image


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def rotate(x, degree):
    return x.rotate(degree)


def gaussian(x, radius):
    return x.filter(MyGaussianBlur(radius))


def enh_bri(x, brightness):
    bri = ImageEnhance.Brightness(x)
    return bri.enhance(brightness)


def enh_sha(x, sharpness):
    sha = ImageEnhance.Sharpness(x)
    return sha.enhance(sharpness)


def enh_con(x, contrast):
    con = ImageEnhance.Contrast(x)
    return con.enhance(contrast)


def enh_col(x, color):
    con = ImageEnhance.Color(x)
    return con.enhance(color)


def scale(x, scale_num):
    return x.resize((int(x.size[0] * scale_num), int(x.size[1] * scale_num)))


mr = [scale, rotate, enh_con, enh_col, enh_bri, enh_sha, gaussian]
par = [0.8, 2, 0.8, 0.8, 0.8, 0.8, 1]
mr_name = ['scale', 'rotation', 'contrast', 'saturation', 'brightness', 'sharp', 'gaussian']


def img_followup(source_path, target_path):
    imgs_name = os.listdir(source_path)
    imgs_followup_name = []
    for img_name in imgs_name:
        img = Image.open(os.path.join(source_path, img_name))
        for i in range(len(mr)):
            img_new = mr[i](img, par[i])
            img_new_name = '.'.join(img_name.split('.')[:-1]) + '_' + mr_name[i] + '.' + img_name.split('.')[-1]
            img_new.save(os.path.join(target_path, img_new_name))
            imgs_followup_name.append(img_new_name)
    return imgs_followup_name


def img_followup_list(imgs_name, source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        imgs_followup_name = []
        for img_name in imgs_name:
            img = Image.open(os.path.join(source_path, img_name))
            for i in range(len(mr)):
                try:
                    img_new = mr[i](img, par[i])
                except:
                    img_new = mr[i](img.convert("RGBA"), par[i])
                # use png to avoid compression
                img_new_name = '.'.join(img_name.split('.')[:-1]) + '_' + mr_name[i] + '.png'
                if not os.path.exists(os.path.dirname(os.path.join(target_path, img_new_name))):
                    os.makedirs(os.path.dirname(os.path.join(target_path, img_new_name)))
                img_new.save(os.path.join(target_path, img_new_name))
                imgs_followup_name.append(img_new_name)
        return imgs_followup_name
    else:
        imgs_followup_name = []
        for img_name in imgs_name:
            for i in range(len(mr)):
                img_new_name = '.'.join(img_name.split('.')[:-1]) + '_' + mr_name[i] + '.' + img_name.split('.')[-1]
                imgs_followup_name.append(img_new_name)
        return imgs_followup_name
