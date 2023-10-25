import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import imageio

# Image directory
img_dir = 'imgs_wmasks'
gif_dir = 'masks\\'

# Iterate over files in image directory
for img_name in os.listdir(img_dir):
    jpg_name = os.path.join(img_dir, img_name)

    # Import image as grayscale array
    img = Image.open(jpg_name).convert('L')
    gray_arr = numpy.array(img)
    
    # # Plot gray
    # plt.imshow(gray_arr, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    
    # Convert to binary array
    thresh_min = 50
    thresh_max = 250
    bin_arr = (gray_arr >= thresh_min) & (gray_arr <= thresh_max)
    bin_arr[200:, :] = 0
    bin_arr = bin_arr.astype('uint8')*255
    
    # # Plot binary
    # plt.imshow(bin_arr, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    
    # Save as gif
    gif_name = gif_dir + img_name[:16] + img_name[19:23] + '.gif'
    imageio.imsave(gif_name, bin_arr)
