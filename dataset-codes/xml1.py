import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
n=0
for id in range(0,17907):
    path="/home/fereshteh/kaist"
    address = os.path.join(path, 'panno', 'I'+  '{0:05}'.format(id) + '.xml')
    imagepath=os.path.join(path, 'person', 'I'+ '{0:05}'.format(id)+ '.png')
    tree = ET.parse(address)
    root = tree.getroot()

    i=0
    for child in root:
    # print(child.tag, child.attrib)
        i=i+1
    

    print(n)
    for j in range (5,i):
    # original_image = Image.open(imagepath, mode='r')
    # image = original_image.convert('RGB')
    # image = normalize(to_tensor(resize(original_image)))
    # image = image.to(device)
        # image = Image.open(imagepath)
        
        tree = ET.parse(address)
        root = tree.getroot()
    # print(root[5][1][1].text)
        # text=root[6].text

        x=root[j][1][0].text
        y=root[j][1][1].text
    # print(y)
        # print(root[5][1].text)
        # k=root[6][0].text
        xmin=root[j][1][0].text
        ymin=root[j][1][1].text
        w=root[j][1][2].text
        if (w==0):
            w=1
        h=root[j][1][3].text
        if (h==0):
            h=1
        xmax=int(xmin)+int(w)
        ymax=int(ymin)+int(h)
        if (int(xmin)<150):
            xnew=xmin
            left=0
        else:
            xnew=150
            left=int(xmin)-150
        if (int(ymin)<150):
            ynew=ymin
            top=0
        else:
            ynew=150
            top=int(ymin)-150
        right=int(xmax)+150
        bottom=int(ymax)+150
        image = Image.open(imagepath)
        image=image.crop((left,top,right,bottom))
        image.save('/home/fereshteh/kaist/augm/rgb/'+ 'I'+ '{0:05}'.format(n)+ '.png')
    # print(xmin,ymin,w,h)
        root[j][1][0].text = root[j][1][0].text.replace(x,str(xnew))
        root[j][1][1].text = root[j][1][1].text.replace(y,str(ynew))
        # xr=root[j][1][0].text
        # yr=root[j][1][1].text
        # wr=root[j][1][2].text
        # hr=root[j][1][3].text
        # shape = [(int(xr), int(yr)), (int(xr)+int(wr),int(yr)+int(hr))] 
        # draw = ImageDraw.Draw(image)
        # draw.rectangle(shape, outline=(255, 255, 255))
        # image.save("/home/fereshteh/kaist/"+str(n)+".jpg", quality=95)
        for m in range(5,j):
            k=root[m][0].text
            if (m!=j):
                root[m][0].text = root[m][0].text.replace(k,'nonperson')
                x1=root[m][1][0].text
                y1=root[m][1][1].text
                w1=root[m][1][2].text
                h1=root[m][1][3].text
                root[m][1][0].text = root[m][1][0].text.replace(x1,'10')
                root[m][1][1].text = root[m][1][1].text.replace(y1,'10')
                root[m][1][2].text = root[m][1][2].text.replace(w1,'50')
                root[m][1][3].text = root[m][1][3].text.replace(h1,'50')
        for m in range(j,i):
            k=root[m][0].text
            if (m!=j):
                root[m][0].text = root[m][0].text.replace(k,'nonperson')
                x1=root[m][1][0].text
                y1=root[m][1][1].text
                w1=root[m][1][2].text
                h1=root[m][1][3].text
                root[m][1][0].text = root[m][1][0].text.replace(x1,'10')
                root[m][1][1].text = root[m][1][1].text.replace(y1,'10')
                root[m][1][2].text = root[m][1][2].text.replace(w1,'50')
                root[m][1][3].text = root[m][1][3].text.replace(h1,'50')
    # root[6][0].text = root[6][0].text.replace(k,'nonperson')
        tree.write('/home/fereshteh/kaist/augm/xml/'+'I'+ '{0:05}'.format(n)+ '.xml', encoding='latin-1')
        n=n+1
    
    