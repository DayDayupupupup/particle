import csv
import math

out = open('sqrtData.csv', 'w', newline='')
csv_writer = csv.writer(out)
label = ['posX','posY','magX','magY','magZ']
csv_writer.writerow(label)

with open('avrData.csv','r') as csvfile:
    data = csv.DictReader(csvfile)
    i = 0.0
    squareMagX = 0.0
    squareMagY = 0.0
    squareMagZ = 0.0
    sqrtMag = 0.0
    for row in data:

        squareMagX = math.pow(float(row['magX']),2)
        squareMagY = math.pow(float(row['magY']),2)
        squareMagZ = math.pow(float(row['magZ']),2)
        sqrtMag = math.sqrt(squareMagX+squareMagY+squareMagZ)

            #print('posX: '+row['posX']+',posY: '+row['posY']+',magX: '+str(avrMagX)+',magY: '+str(avrMagY)+',magZ: '+str(avrMagZ))\
        list = [row['posX'],row['posY'],str(sqrtMag)]
        csv_writer.writerow(list)