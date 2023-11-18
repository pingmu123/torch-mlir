def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time # seconds
    elapsed_mins = int(elapsed_time / 60) # minutes
    # total time: xxx mins and yyy secs
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs