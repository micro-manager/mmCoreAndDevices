from skimage import data

def dummyscan():
    camera = data.camera()

    return camera


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(dummyscan())
    plt.show()