import glob

imgs = glob.glob("images/*/*.png")

print(len(imgs))

import os

for img in imgs:
    code = f"convert {img} {img[:-4]}.jpg"
    print(code)
    os.system(code)