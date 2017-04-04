import os

f = open('result_standard.csv')

for line in f:
    line_parts = line.split(',')
    name = line_parts[0]

    try:
        lag_prob = float(line_parts[8])

        if lag_prob > 0.5:
            os.system('cp ../test\ dataset/' + name + ' /home/xyh3984/')

    except:
        pass
