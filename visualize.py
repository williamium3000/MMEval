from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import argparse

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

coco17_instance = COCO("data/coco/annotations/instances_val2017.json")
coco17_caption = COCO("data/coco/annotations/captions_val2017.json")
cats17 = coco17_instance.loadCats(coco17_instance.getCatIds())
id_name_mapping17 = {cat["id"]: cat["name"] for cat in cats17}
coco14_instance = COCO("data/coco/annotations/instances_val2014.json")
coco14_caption = COCO("data/coco/annotations/captions_val2014.json")
cats14 = coco14_instance.loadCats(coco14_instance.getCatIds())
id_name_mapping14 = {cat["id"]: cat["name"] for cat in cats14}

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int)
args = parser.parse_args()

img = coco17_instance.loadImgs([args.id])[0]
I = io.imread(img['coco_url'])
plt.imshow(I)
plt.axis('off')
# annIds = coco17_instance.getAnnIds(imgIds=img['id'], iscrowd=None)
# anns = coco17_instance.loadAnns(annIds)
# coco17_instance.showAnns(anns, True)
plt.savefig(f"{args.id}.png")