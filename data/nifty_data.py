import shutil
import glob
import csv
import dateutil.parser
import pdb
from collections import OrderedDict

outfilename = "all_data.csv"
with open(outfilename, 'wb') as outfile:
	#writer = csv.writer(outfile)
	#writer.writerow(["Index", "Date","Time","Open","Close","Low","High","Volume Traded"])
	for filename in glob.glob('*.txt'):
		if filename == outfilename:
			# don't want to copy the output into the output
			continue
		with open(filename, 'rb') as readfile:
			shutil.copyfileobj(readfile, outfile)


reader = csv.reader(open(outfilename, 'r'))
data_dict = {}
for row in reader:
	print(row)
	date_time = dateutil.parser.parse(row[1] + "-" + row[2])
	if len(row) < 8:
		row.append('0')
	data_dict[date_time] = row
ordered = OrderedDict(sorted(data_dict.items(), key=lambda t:t[0]))
with open("sorted_data.csv", 'wb') as outfile:
	writer = csv.writer(outfile)
	writer.writerow(["Index","DateTime","Open","Close","Low","High","Volume Traded"])
	for row in ordered.items():
		k = row[0]
		v = row[1]
		writer.writerow([k,"Nifty",v[3],v[4],v[5],v[6],v[7]])