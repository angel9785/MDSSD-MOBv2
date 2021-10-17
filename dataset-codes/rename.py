import os
# Function to rename multiple files
def main():
   i =23036
   path="//home/fereshteh/kaist/person_sani_lw/"
   path1="/home/fereshteh/kaist/sanitized_lw/"
   for filename in sorted(os.listdir(path)):

      my_dest ='I'+ format(i, '05d') + ".png"
      print(filename+" to "+my_dest)
      my_source =path + filename
      my_dest =path1 + my_dest

  # rename() function will
  # rename all the files
      os.rename(my_source, my_dest)
      i += 1
      print(i)

# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()
