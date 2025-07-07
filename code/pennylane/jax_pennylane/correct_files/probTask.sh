#!/bin/bash

for p in $(seq 6 1 10)
do
  /home/francesco/PycharmProjects/qaoa_transferability/.venv/bin/python probaTask_randomGenerated_acceptor.py $p

done