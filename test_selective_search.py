from skimage import io
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

img = io.imread('../train dataset/ALB/img_00043.jpg')


img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=500)

candidates = set()
for r in regions:
    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 2000:
        continue

    # distorted rects
    x, y, w, h = r['rect']

    if w / h > 1.2 or h / w > 1.2:
        continue

    if r['size'] > img.shape[0] * img.shape[1] / 3:
        continue

    if h >= img.shape[0]/3 or w >= img.shape[1]/3:
        continue

    candidates.add(r['rect'])

# draw rectangles on the original image
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)
for x, y, w, h in candidates:
    print x, y, w, h
    rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

plt.show()
