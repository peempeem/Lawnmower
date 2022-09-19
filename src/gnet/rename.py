import os

def rename(img_dir, mask_dir, base=""):
    images = os.listdir(img_dir)
    for i in range(len(images)):
        parts = images[i].split('.')
        image_path = f"{img_dir}/{images[i]}"
        mask_path = f"{mask_dir}/{parts[0]}.png"
        image_out = f"{img_dir}/{base} {i}.{parts[1]}"
        mask_out = f"{mask_dir}/{base} {i}.png"
        os.rename(image_path, image_out)
        os.rename(mask_path, mask_out)

if __name__ == '__main__':
    image_dir = "../images"
    masks_dir = "../masks"
    rename(image_dir, masks_dir, "b3")
