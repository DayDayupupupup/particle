import csv

out = open('avrData.csv', 'w', newline='')
csv_writer = csv.writer(out)
label = ['posX','posY','magX','magY','magZ']
csv_writer.writerow(label)

with open('data.csv','r') as csvfile:
    data = csv.DictReader(csvfile)
    i = 0.0
    avrMagX = 0.0
    avrMagY = 0.0
    avrMagZ = 0.0
    avrPoseX = 0.0
    avrPoseY = 0.0

    for row in data:
        i = i + 1.0
        avrMagX = avrMagX + float(row['magX'])
        avrMagY = avrMagY + float(row['magY'])
        avrMagZ = avrMagZ + float(row['magZ'])
        if i == 6:
            avrMagX = avrMagX / 6
            avrMagY = avrMagY / 6
            avrMagZ = avrMagZ / 6
            i = 0

            #print('posX: '+row['posX']+',posY: '+row['posY']+',magX: '+str(avrMagX)+',magY: '+str(avrMagY)+',magZ: '+str(avrMagZ))\
            list = [row['posX'],row['posY'],str(avrMagX),str(avrMagY),str(avrMagZ)]
            csv_writer.writerow(list)







