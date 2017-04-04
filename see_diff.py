import csv
import numpy
import os

standard_file = 'result_standard.csv'
test_file = 'result_revised.csv'

all_types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

standard_lines = [l for l in csv.reader(open(standard_file))]
test_lines = [l for l in csv.reader(open(test_file))]

total_miss = 0
alb_miss = 0
bet_miss = 0
dol_miss = 0
lag_miss = 0
other_miss = 0
shark_miss = 0
yft_miss = 0
for standard_line in standard_lines:
    for test_line in test_lines:
        standard_name = standard_line[0]
        test_name = test_line[0]

        if test_name == 'image':
            continue

        if standard_name == test_name:
            test_probs = [float(i) for i in test_line[1:len(test_line)]]
            standard_probs = [float(i) for i in standard_line[1:len(standard_line)]]

            if numpy.argmax(test_probs) != numpy.argmax(standard_probs):
                test_index = numpy.argmax(test_probs)
                standard_index = numpy.argmax(standard_probs)

                if standard_index == 4:
                    continue

                if test_index == 0:
                    alb_miss += 1

                if test_index == 1:
                    bet_miss += 1

                if test_index == 2:
                    dol_miss += 1

                if test_index == 3:
                    lag_miss += 1

                if test_index == 5:
                    other_miss += 1

                if test_index == 6:
                    shark_miss += 1

                if test_index == 7:
                    yft_miss += 1

                total_miss += 1

                print 'stand:'+all_types[standard_index]+'------>'+'predict:'+all_types[test_index]

print total_miss

print alb_miss, bet_miss, dol_miss, lag_miss, other_miss, shark_miss, yft_miss
