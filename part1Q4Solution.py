from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

dir = "./data/train"
mean_images={}
for c in os.listdir(dir):
    this_dir = os.path.join(dir,c)
    if not os.path.isdir(this_dir): continue
    mean_images[c]=0
    files = os.listdir(this_dir)
    count=0
    for file in files :
        if ".jpg" not in file: continue
        file_path = os.path.join(this_dir,file)
        img = Image.open(file_path)
        mean_images[c]+=np.array(img)/255
        count+=1
    mean_images[c] = mean_images[c]/count
for i,c in enumerate(mean_images):
    plt.subplot(3,4,i+1)
    plt.imshow(mean_images[c])
    plt.axis("Off")
    plt.title(c)
plt.tight_layout()
plt.savefig("class_means.pdf")
plt.show()