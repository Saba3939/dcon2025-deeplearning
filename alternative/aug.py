sensor_result = []
class_result = []
goodness_result = []

with open("sensor_data.csv", encoding='utf-8') as sensor_data, \
     open("class_data.csv", encoding='utf-8') as class_data, \
     open("goodness_data.csv", encoding='utf-8') as goodness_data :
    for i in sensor_data.read().split("\n") :
        if i != "" :
            sensor_result.append(i.split(","))
    for i in class_data.read().split("\n") :
        if i != "" :
            class_result.append(i)
    for i in goodness_data.read().split("\n") :
        if i != "" :
            goodness_result.append(i)


print(sensor_result[0])

for i in range(sensor_result.__len__()) :
    for j in range(9) :
        sensor_result[i][j] = str(float(sensor_result[i][j]) -0.1)

print(sensor_result[0])


with open("alt_sensor_data.csv", mode='a', newline='', encoding='utf-8') as sensor_data, \
     open("alt_class_data.csv", mode='a', newline='', encoding='utf-8') as class_data, \
     open("alt_goodness_data.csv", mode='a', newline='', encoding='utf-8') as goodness_data :
    for i in sensor_result :
        sensor_data.write(",".join(i))
        sensor_data.write("\n")
    for i in class_result :
        class_data.write(i)
        class_data.write("\n")
    for i in goodness_result :
        goodness_data.write(i)
        goodness_data.write("\n")