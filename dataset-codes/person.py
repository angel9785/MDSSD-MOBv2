import os


# Function to rename multiple files
def main():
    i = 0
    j=0
    path = "/home/fereshteh/kaist/txt/"
    path1 = "/home/fereshteh/kaist/rgb/"
    path2 = "/home/fereshteh/kaist/person/"
    path3 = "/home/fereshteh/kaist/anno/"
    path4 = "/home/fereshteh/kaist/panno/"
    path5 = "/home/fereshteh/kaist/lw/"
    path6 = "/home/fereshteh/kaist/plw/"
    for filename in sorted(os.listdir(path)):
        f = open(path + filename, "r")
        s1 = f.readline()
        s2 = f.readline()

        if s2.find("person") != -1:
            my_dest1 = 'I' + format(j, '05d') + ".png"
            my_src1 = 'I' + format(i, '05d') + ".png"
            my_dest2 = 'I' + format(j, '05d') + ".xml"
            my_src2 = 'I' + format(i, '05d') + ".xml"

            print(my_src1 + " to " + my_dest1)
            my_source1 = path1 + my_src1
            my_dst1 = path2 + my_dest1
            my_source2 = path3 + my_src2
            my_dest2 = path4 + my_dest2
            my_source3 = path5 + my_src1
            my_dest3 = path6 + my_dest1
            os.rename(my_source1, my_dst1)
            os.rename(my_source2, my_dest2)
            os.rename(my_source3, my_dest3)
            j=j+1

        i=i+1


# rename() function will
# rename all the files

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
