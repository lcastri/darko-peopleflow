#!/bin/bash

bagname="06-02-2025-DARKO"
step=20
for i in {0..20}
do
    if [ $i -eq 0 ]; then
        load_goal=false
        start_time=0
    else
        load_goal=true
        start_time=$(($step * (i-1) - 20))
    fi

    if [ $i -eq 20 ]; then
        time_of_the_day="off"
    else
        time_of_the_day="T$i"
    fi
    roslaunch bag_postprocess bringup.launch bagname:=$bagname time_of_the_day:=$time_of_the_day start_time:=$start_time load_goal:=$load_goal
done
