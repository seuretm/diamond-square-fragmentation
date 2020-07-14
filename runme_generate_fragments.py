# This code first fragment sample.jpg using the diamond-square approach,
# and fragments it a second time using the rectangular approach.

from scipy.misc import imresize
from PIL import Image
import numpy
from scipy.ndimage.measurements import label
#import sys
from horizon import horizon
from PIL import ImageChops
from diamondsquare import diamond_square
from torchvision import transforms
#from tqdm import tqdm
#import sys
#import os

color_trans = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.000)

image = Image.open('sample.jpg')

# Generate the diamond square image
small_size = 768
ds = diamond_square(desired_size=small_size, min_height=0, max_height=255, roughness=0.65, AS_NP_ARRAY=True, chaos=1)
desired_size = max(image.size)
ds = imresize(ds, (desired_size, desired_size))
ds = ds[0:image.size[1], 0:image.size[0]]

# Save the diamond square image
Image.fromarray((ds).astype('uint8')).convert('RGB').save('0-diamond-square.png')

# Binarize the diamond square
margin = 10
nb_pixels = ds.shape[0] * ds.shape[1]
mean = numpy.mean(ds)
binary = numpy.array((ds<mean-margin) + (ds>mean+margin), dtype=numpy.int)

# take care of side borders
ds_h = binary.shape[0]
ds_w = binary.shape[1]
horizon_roughness = 0.65
b = horizon(binary.shape[0], horizon_roughness, ds_h//20)
for i in range(binary.shape[0]):
    binary[i, 0:b[i]] = 0
b = horizon(binary.shape[0], horizon_roughness, ds_h//20)
for i in range(binary.shape[0]):
    binary[i, (binary.shape[1]-b[-i]):binary.shape[1]] = 0
b = horizon(binary.shape[1], horizon_roughness, ds_w//20)
for i in range(binary.shape[1]):
    binary[0:b[i], i] = 0
b = horizon(binary.shape[1], horizon_roughness, ds_w//20)
for i in range(binary.shape[1]):
    binary[(binary.shape[0]-b[-i]):binary.shape[0], i] = 0

# Save it for fun
Image.fromarray((255*binary).astype('uint8')).convert('RGB').save('1-binary.png')

# select connected components...
mask = numpy.array([[0,1,0],[1,1,1],[0,1,0]])
min_frag_size = 0.025
labeled, ncomponents = label(binary, mask)
frag_num = 0
for c in range(1, numpy.max(labeled)+1):
    frag = labeled==c
    surface = numpy.sum(frag)
    if surface>=min_frag_size * nb_pixels: # ... that are big enough
        # visualization of the binary fragment
        im = Image.fromarray((255*frag).astype('uint8')).convert('RGB')
        im.save('2-fragment-%d.png' % frag_num)
        # creation of the RGB fragment
        bbox = im.getbbox()
        res = ImageChops.multiply(im, image).crop(bbox)
        shape_trans = transforms.Compose([
            transforms.Pad(max(res.size)//3),
            transforms.RandomAffine(1, scale=(0.99,1.01), fillcolor=0)
        ])
        res = shape_trans(res)
        bbox = res.getbbox()
        res = color_trans(res.crop(bbox))
        res.save('3-fragment-%d.jpg' % frag_num, quality=99)
        frag_num += 1


im = Image.open('sample.jpg')
in_w = im.size[0]
in_h = im.size[1]

mask = numpy.array([[0,1,0],[1,1,1],[0,1,0]])

arr = 255*numpy.ones([im.size[1], im.size[0]])

# side borders
b = horizon(arr.shape[0], 0.9, 10)
for i in range(arr.shape[0]):
    arr[i, 0:b[i]] = 0
b = horizon(arr.shape[0], 0.9, 10)
for i in range(arr.shape[0]):
    arr[i, (arr.shape[1]-b[-i]):arr.shape[1]] = 0
b = horizon(arr.shape[1], 0.9, 10)
for i in range(arr.shape[1]):
    arr[0:b[i], i] = 0
b = horizon(arr.shape[1], 0.9, 10)
for i in range(arr.shape[1]):
    arr[(arr.shape[0]-b[-i]):arr.shape[0], i] = 0

# vertical cuts
for cut in range(1, 3):
    h1 = horizon(arr.shape[0], 0.6, arr.shape[1]//10)
    h2 = horizon(arr.shape[0], 0.5, 10)
    h3 = horizon(arr.shape[0], 0.8, arr.shape[1]//25)
    h = h1+h2+cut*arr.shape[1]//3
    for i in range(arr.shape[0]):
        j = (i+1) % arr.shape[0]
        a = min(h[i], h[j]) - h3[i]//2
        b = max(h[i], h[j]) + h3[i]//2 + 2
        arr[i, a:b] = 0

# horizontal cuts
for cut in range(1, 3):
    h1 = horizon(arr.shape[1], 0.6, arr.shape[1]//10)
    h2 = horizon(arr.shape[1], 0.5, 10)
    h3 = horizon(arr.shape[1], 0.8, arr.shape[1]//25)
    h = h1+h2+cut*arr.shape[0]//3
    for i in range(arr.shape[1]):
        j = (i+1) % arr.shape[1]
        a = min(h[i], h[j]) - h3[i]//2
        b = max(h[i], h[j]) + h3[i]//2 + 2
        arr[a:b, i] = 0

Image.fromarray(arr.astype('uint8')).convert('RGB').save('4-grid.png')

labeled, ncomponents = label(arr, mask)

nb_pixels = arr.shape[0] * arr.shape[1]
mx = numpy.max(labeled)+1
high_id = 0
for c in range(1, mx):
    binary = labeled==c
    s = numpy.sum(binary)
    if s<nb_pixels/27:
        continue
    fmask = Image.fromarray((255*binary).astype('uint8')).convert('RGB')
    fmask.save('5-fragment-%d.png' % high_id)
    
    bbox = fmask.getbbox()
    res = ImageChops.multiply(fmask, im).crop(bbox)
    shape_trans = transforms.Compose([
        transforms.Pad(max(res.size)//3),
        transforms.RandomAffine(1, scale=(0.99,1.01), fillcolor=0)
    ])
    res = shape_trans(res)
    bbox = res.getbbox()
    res = color_trans(res.crop(bbox)            )
    res.save('6-fragment-%d.jpg' % high_id, quality=99)
    high_id += 1
