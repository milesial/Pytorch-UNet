from PIL import Image
import numpy
import matplotlib.pyplot as plt
import imageio
 
# Import image as grayscale array
img= Image.open("BPH009_mask+000056-000.jpg").convert('L')
gray_arr = numpy.array(img)

# Convert to binary array
thresh_min = 50
thresh_max = 70
bin_arr = (gray_arr >= thresh_min) & (gray_arr <= thresh_max)
bin_arr = bin_arr.astype('uint8')*255
bin_arr[200:, :] = 0

# Plot
plt.imshow(bin_arr, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

# Save as gif
imageio.imsave("mask.gif", bin_arr)
