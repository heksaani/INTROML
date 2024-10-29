# Note 2

# Posterization

code from here : [https://github.com/kevinioi/posterizer-kmeans/](https://github.com/kevinioi/posterizer-kmeans/blob/master/posterizer/posterize.py)

```python
def posterize(original_img, degree):
    '''
        Applies k-means clustering to find the dominant colours
        in an image and resets all pixels to be the closest dominant
        colour

        params:
            original_img (array or image): image to be altered
            degree(int): degree to posterize the image,
                lower value means higher level of alteration
        return:
            PIL image:
                posterized image
    '''
    if not isinstance(original_img,np.ndarray):
        np_img = np.array(original_img)
    else:
        np_img = original_img

    dims = np_img.shape
    np_img.shape = (np.prod(dims[:-1]),dims[-1])

    model = KMeans(n_clusters=degree,random_state=101)
    model.fit(np_img)
    centroids = model.cluster_centers_

    for idx, pixel_label in enumerate(model.labels_):
        np_img[idx] = centroids[pixel_label]
    np_img.shape = dims

    return Image.fromarray(np_img)

def pool_smoothing(img, kernel_size=(2, 2), stride=(1, 1), method='avg'):
    """
    Applies kernel-based smoothing (average, min, or max pooling) to an image.
    
    Parameters:
        img (numpy array): Image in NumPy array format.
        kernel_size (tuple): Size of the kernel (height, width).
        stride (tuple): Stride of the kernel (step size in y and x).
        method (str): Pooling method ('avg', 'min', 'max').
    """
    dims = img.shape

    if len(dims) > 3:
        raise ValueError("Cannot pool img with dimensions {}".format(dims))

    if method == 'avg':
        pool_func = lambda x: np.mean(np.mean(x, axis=0), axis=0)
    elif method == 'max':
        pool_func = lambda x: np.max(np.max(x, axis=0), axis=0)
    elif method == 'min':
        pool_func = lambda x: np.min(np.min(x, axis=0), axis=0)
    else:
        raise ValueError("Invalid pooling method provided: {}".format(method))

    smoothed_img = np.copy(img)

    for idy in range(0, dims[0] - kernel_size[0] + 1, stride[0]):
        for idx in range(0, dims[1] - kernel_size[1] + 1, stride[1]):
            smoothed = pool_func(img[idy:idy + kernel_size[0], idx:idx + kernel_size[1]])
            smoothed_img[idy:idy + kernel_size[0], idx:idx + kernel_size[1]] = smoothed

    return smoothed_img
```
