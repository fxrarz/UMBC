# parameter count

flat_count = 4.44*10000
stairway_count = 2.98*100000
model_count = 1.11*100000

total_count = flat_count + stairway_count

if model_count < total_count:
	reduction = model_count/total_count
else:
	reduction = total_count/model_count

reduction_percentage = reduction * 100
print(reduction_percentage)


# disk size

flat_size = 0.17
stairway_size = 1.138
model_size = 0.425

total_size = flat_size + stairway_size

diff = abs(total_size - model_size)

reduction_percentage = abs(diff/total_size) * 100
print(reduction_percentage)


# parameter count

flat_count = 4.44*10000
stairway_count = 2.98*100000
model_count = 1.11*100000

total_count = flat_count + stairway_count

diff = abs(total_count - model_count)

reduction_percentage = abs(diff/total_count) * 100
print(reduction_percentage)


# cpu time

flat_time = .14
stairway_time = 1.07
model_time = 0.2

total_time = flat_time + stairway_time

diff = abs(total_time - model_time)

reduction_percentage = abs(diff/total_time) * 100
print(reduction_percentage)


# gpu time

flat_time = .15
stairway_time = 0.13
model_time = 0.14

total_time = flat_time + stairway_time

diff = abs(total_time - model_time)

reduction_percentage = abs(diff/total_time) * 100
print(reduction_percentage)

