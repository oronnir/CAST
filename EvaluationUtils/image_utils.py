from PIL import Image


def crop_image(image_path, x, y, w, h):
    cropped_im = None
    try:
        im = Image.open(image_path)
        area = (x, y, x+w, y+h)
        cropped_im = im.crop(area)
    except Exception as e:
        print('cropping image: {} failed with exception: {}'.format(image_path, str(e)))
    return cropped_im


def save_image(image_path, image_bytes):
    if image_bytes:
        image_bytes.save(image_path)
    return
