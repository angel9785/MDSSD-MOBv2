import re
import os
j=0
path="/home/fereshteh/kaist/sanitized_annotations/"
for filename in sorted(os.listdir(path)):
    
    filename3=filename
    filename3 = re.sub('.txt',".png",str(filename3))
    filename2 = re.sub('_',"/",str(filename))
    filename = re.sub('/I',"/visible/I",str(filename2))
    filename = re.sub('.txt',".jpg",str(filename))
    filename1=re.sub('/V000/visible/I[0-9].jpg',"/images/",str(filename))
    filename1 = filename2[:5]+filename2[6:].replace(filename2[6:], "")+"/images/"
    filename=str("/home/fereshteh/kaist/test/")+str(filename1)+str(filename)
    # print(filename)
    # my_dst="/home/fereshteh/kaist/sanitest/"+'I' + format(j, '05d') + ".png"
    # filename = re.sub('.jpg',".png",str(filename))
    my_dst="/home/fereshteh/kaist/sanitized_annotations1/sanitized/"+str(filename3)
    os.rename(filename, my_dst)
    j=j+1
    print(j)
