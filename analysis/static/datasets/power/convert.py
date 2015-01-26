from datetime import date
import fileinput
import re
import sys

d0 = date(2006, 1, 1)

# Extracts the date and time information and converts it into a day integer.
for line in fileinput.input():
	if "?" in line:
		continue	# Skip lines with missing information.
	result = re.match("([0-9]+)/([0-9]+)/([0-9]+);([0-9]+):([0-9]+):([0-9]+);(.+)", line)
	d1 = date(int(result.groups()[2]), int(result.groups()[1]), int(result.groups()[0]))
	days = (d1-d0).days
	seconds = int(result.groups()[3])*60*60 + int(result.groups()[4])*60 + int(result.groups()[5])
	print "%i;%i;%s" % (days, seconds, result.groups()[6])
