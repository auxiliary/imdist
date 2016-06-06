#!/usr/bin/env python

from PIL import Image
import sys

src = sys.argv[1]
dst = sys.argv[2]

img = Image.open(open(src))
img = img.convert('L')
img.save(dst)
