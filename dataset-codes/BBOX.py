import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
path="/home/fereshteh/kaist"
id=1500
address = os.path.join(path, 'panno', 'I'+  '{0:05}'.format(id) + '.xml')
imagepath=os.path.join(path, 'person', 'I'+ '{0:05}'.format(id)+ '.png')
tree = ET.parse(address)
root = tree.getroot()
xc=root[5][1][0].text
yc=root[5][1][1].text
    # print(y)
        # print(root[5][1].text)
        # k=root[6][0].text

w=root[5][1][2].text
if (w==0):
    w=1
h=root[5][1][3].text
if (h==0):
    h=1
# xmin=root[5][1][0].text
# ymin=root[5][1][1].text
xmin=int(xc)-(int(w)/2)
ymin=int(yc)-(int(h)/2)
xmax=int(xc)+(int(w)/2)
ymax=int(yc)+(int(h)/2)
image = Image.open(imagepath)
shape = [(int(xmin), int(ymin)), (xmax,ymax)] 
draw = ImageDraw.Draw(image)
draw.rectangle(shape, outline=(255, 255, 255))
image.save("/home/fereshteh/kaist/"+str(id+1)+".jpg", quality=95)