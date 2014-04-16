import csv
import sys

if (len(sys.argv) < 2):
    print "Usage: %s <csv_file> (<error>)" % sys.argv[0]
    sys.exit()

input_file = sys.argv[1]

ifile  = open(input_file, "rb")
reader = csv.reader(ifile, delimiter=";")

header = True
selected_col = 0

sum = 0.0
row_count = 0
error = "Error"

for row in reader:
    if header:
        for col in row:
            if (col.strip() != error):
                selected_col += 1
            else:
                break
        if (selected_col == len(row)):
            print "Error-type %s not present in given file!" % error
            sys.exit()
        header = False
    else:
        sum += float(row[7])
        row_count += 1

print("%s: %f" % (input_file,(sum / float(row_count))))
