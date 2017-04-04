import csv

target_csv_filepath = 'result.csv'
save_csv_filepath = 'result_revised.csv'

writer = csv.writer(open(save_csv_filepath, 'wb'))
writer.writerow(['image', 'ALB'	, 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])


top1_value = 0.9
top2_value = 0.2
top3_value = 0.1
top4_value = 0.06
top5_value = 0.03
top6_value = 0.03
top7_value = 0.03
top8_value = 0.03

small_to_large_values = [top8_value, top7_value, top6_value, top5_value, top4_value, top3_value,
                         top2_value, top1_value]


f = open(target_csv_filepath)

for line in f:


        image_results = line.replace('\n', '').split(',')
        image_name = image_results[0]
        if image_name == 'image':
            continue
        image_name = image_name.replace('+AF8-', '_')
        print image_name
        image_results = image_results[1: len(image_results)]
        image_results = [float(i.replace('+AC0', '')) for i in image_results]

        if image_results[0] > 0.5 and image_results[0] < 0.6:
            #image_results = [0.5, 0.5, 0.03, 0.03, 0.03, 0.03, 0.03, 0.6]
            pass
        elif image_results[7] > 0.3 and image_results[7]<0.9999:
            image_results = [0.8, 0.4, 0.03, 0.03, 0.03, 0.03, 0.03, 0.3]

        elif image_results[5] > 0.3 and image_results[5] < 0.999:
            image_results = [0.7, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.3]

        elif image_results[6] > 0.3 and image_results[6] < 0.999:
            image_results = [0.8, 0.4, 0.03, 0.03, 0.03, 0.03, 0.03, 0.3]
            
        if image_results[1] > 0.3:
            image_results = [0.6, 0.7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

        else:
            small_to_big_indices = sorted(range(len(image_results)), key=lambda i: image_results[i])[-8:]
            i = 0
            
            for index in small_to_big_indices:
                image_results[index] = small_to_large_values[i]
                i += 1

        image_results = [image_name] + image_results

        writer.writerow(image_results)

f.close()
