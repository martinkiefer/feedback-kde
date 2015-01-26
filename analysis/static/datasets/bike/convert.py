from datetime import date
import fileinput
import re
import sys

d0 = date(2006, 1, 1)

# Extracts the date and time information and converts it into a day integer.
for line in fileinput.input():
	result = re.match("[0-9]+,([0-9]+)-([0-9]+)-([0-9]+),(.+)", line)
	d1 = date(int(result.groups()[0]), int(result.groups()[1]), int(result.groups()[2]))
	days = (d1-d0).days
	print "%i,%s" % (days, result.groups()[3])
