import sys
import os

filelist = os.listdir(sys.argv[1])

path = sys.argv[1].rstrip("/") + "/"
for fname in filelist:
    if fname.endswith('inkml'):
        os.system('crohme2lg %s%s %s%s' % (path, fname, path, fname.rstrip("inkml") + "lg"))
