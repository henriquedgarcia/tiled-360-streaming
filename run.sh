#!/bin/bash

for ((i=0; i<28; i++))
do
  j=$((i + 1))
  for tiling in 12x8 9x6 6x4 3x2 9x6 1x1
    do
      for qlt in 16 22 28 34 40
        do
          python main.py -r 0 0 2 -slice $i $j -tiling $tiling -quality $qlt &
        done
      wait
    done
done


# Lumine
# 0 - angel_falls
# 1 - blue_angels
# 2 - cable_cam
# 3 - chariot_race
# 4 - closet_tour

# Servidor 0
# 5 - drone_chases_car
# 6 - drone_footage
# 7 - drone_video
# 8 - drop_tower
# 9 - dubstep_dance

# Servidor 1
# 10 - elevator_lift
# 11 - glass_elevator
# 12 - montana
# 13 - motorsports_park
# 14 - nyc_drive

# Fortrek
# 15 - pac_man
# 16 - penthouse
# 17 - petite_anse
# 18 - rhinos
# 19 - sunset

# HP-Elite
# 20 - three_peaks
# 21 - video_04
# 22 - video_19
# 23 - video_20
# 24 - video_22

# Alambique
# 25 - video_23
# 26 - video_24
# 27 - wingsuit_dubai

