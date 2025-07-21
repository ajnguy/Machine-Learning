from PIL import Image
import os
import numpy as np

train_dir = 'data/train'
class_data = {}

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        images = []
        for img in os.listdir(class_path):
            image_path = os.path.join(class_path, img)
            image = Image.open(image_path)
            image = np.array(image)
            images.append(image)

        average_image = np.mean(images, axis=0)
        class_data[class_name] = (average_image, len(images))

majority_class = None
max_instances = 0
for class_name, (average_image, num_instances) in class_data.items():
    if num_instances > max_instances:
        max_instances = num_instances
        majority_class = class_name
majority_error_rate = 1 - class_data[majority_class][1] / sum([vals[1] for vals in class_data.values()])

print("Majority Class:", majority_class)
print("Error Rate for Majority Class:", majority_error_rate)

for class_name, (average_image, _) in class_data.items():
    average_image = Image.fromarray(average_image.astype('uint8'))
    average_image.save(f'{class_name}_average.jpg')
