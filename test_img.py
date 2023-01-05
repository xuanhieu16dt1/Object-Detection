from os.path import join
from glob import glob

files = []
for ext in ( '*.png', '*.jpg'):
   files.extend(glob(join("path/to/dir", ext)))

print(files)
