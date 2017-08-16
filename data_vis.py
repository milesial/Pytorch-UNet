import matplotlib.pyplot as plt

def plot_img_mask(img, mask):
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mask)

    plt.show()
