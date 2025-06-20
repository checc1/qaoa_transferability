#!/bin/bash

for i in $(seq 4 2 14)
do
  j=$((i + 2))
  /home/francesco/PycharmProjects/qaoa_transferability/.venv/bin/python scalability_task.py $i $j
  if [ "$i" -eq 14 ]; then
    break
  fi
done