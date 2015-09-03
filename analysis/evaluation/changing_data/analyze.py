import csv
import collections
import sys

f = open(sys.argv[1])
reader = csv.reader(f, delimiter=';')

history=100

error_history = collections.deque()

row_cnt = 0
errorsum = 0
for row in reader:
  if row_cnt == 0: 
    # Skip the header
    row_cnt += 1
    continue
  row_cnt += 1
  errorsum += float(row[1])
  error_history.appendleft(float(row[1]))
  
  if (len(error_history) == history):
    print "%f\t%s" % (errorsum / history, row[8])
    errorsum -= error_history.pop()

f.close()
