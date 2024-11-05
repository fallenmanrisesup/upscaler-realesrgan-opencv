import os
import cv2
from pathlib import Path
from upscaler import Upscaler
import time

cv2.setNumThreads(0)

output_folder = "./data/outputs/"

def store_result(result, label, image_path, is_torch, elapsed_time):
    os.makedirs(output_folder + f"/{label}/", exist_ok=True)
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]

    elapsed_log = f"Elapsed: {elapsed_time:.2f}s"
    with open(f"{output_folder}/{label}/{image_name}.log", "w") as file:
        file.write(elapsed_log)

    if is_torch:
        result.save(f'{output_folder}/{label}/{image_name}.jpg')
        return

    output_path = str(Path(output_folder + f"/{label}") / (image_name + ".jpg"))
    cv2.imwrite(output_path, result)
    return

def process_image(upscaler, label, image_path):
    t = time.time()
    output_image = upscaler.run(label, image_path)
    elapsed_time = time.time() - t
    store_result(output_image['result'], label, image_path, output_image['is_torch'], elapsed_time)
    return

if __name__ == "__main__":
    input_images = ["./data/inputs/04_person_square.jpg"]

    upscaler = Upscaler(
        configs=[
            {
                'reshape': [-1,3,-1,-1],
                'label': 'RealESRGAN_x2plus_dynamic',
                'path': './models/RealESRGAN_x2plus/RealESRGAN_x2plus_dynamic.xml',
                'scale_factor': 2,
                'is_torch': False,
            },
            {
                'reshape': [-1,3,-1,-1],
                'label': 'RealESRGAN_x4plus_dynamic',
                'path': './models/RealESRGAN_x4plus/RealESRGAN_x4plus_dynamic.xml',
                'scale_factor': 4,
                'is_torch': False,
            },
            {
                'reshape': [-1,3,-1,-1],
                'label': 'RealESRGAN_x4plus_torch',
                'path': './models/RealESRGAN_x4plus/RealESRGAN_x4.pth',
                'scale_factor': 4,
                'is_torch': True,
            },
        ],
        device='CPU'
    )

    for path in input_images:
        for label in ['RealESRGAN_x2plus_dynamic', 'RealESRGAN_x4plus_dynamic', 'RealESRGAN_x4plus_torch']:
            process_image(upscaler, label, path)

