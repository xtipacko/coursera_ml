from PIL import Image
import numpy as np
from pprint import pprint
from plot2d import Plot2D
import matplotlib.pyplot as plt
from timeit import default_timer

start = default_timer()

im = Image.open("dps.bmp")
mask_ar = np.array(im)

el_lst = []
for i, row in enumerate(mask_ar):
    for j, element in enumerate(row):
        if not element:
            el_lst.append((j-250, 250-i))

row_datapoints = np.array(el_lst[::100])


t = default_timer() - start
print(f'Data obtained for {t:.3f}s')
