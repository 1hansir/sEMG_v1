import time

start_time = time.time()
count = -1
prev_state = 0


while 1:
    cur_time = time.time()
    time_interval = (cur_time - start_time)//1
    if(time_interval > prev_state):
        print("00:0{}    count:{}".format(time_interval,count))
        prev_state = time_interval
        if(time_interval >= 8):
            count += 1
            start_time = time.time()
            prev_state = 0