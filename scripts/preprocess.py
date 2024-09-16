from PIL import Image
import os

def preprocess_images(raw_dir, processed_dir, scale_factor=2, image_size=(33, 33)):
    raw_dir = os.path.abspath('D:/Development/image-super-resolution/data/raw_images')
    processed_dir = os.path.abspath('D:/Development/image-super-resolution/data/processed_images')

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for filename in os.listdir(raw_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(raw_dir, filename))
            hr = img.convert('RGB')
            lr = hr.resize((hr.width // scale_factor, hr.height // scale_factor), Image.BICUBIC)

            # Resize to a fixed size
            lr = lr.resize(image_size, Image.BICUBIC)
            hr = hr.resize(image_size, Image.BICUBIC)
            
            lr.save(os.path.join(processed_dir, 'LR_' + filename))
            hr.save(os.path.join(processed_dir, 'HR_' + filename))

if __name__ == "__main__":
    raw_dir = '../data/raw_images'
    processed_dir = '../data/processed_images'
    preprocess_images(raw_dir, processed_dir)
