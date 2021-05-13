@echo off
title Test script for testing networks

python src\adas\train.py --config src\adas\configTests\configTest1.yaml
python src\adas\train.py --config src\adas\configTests\configTest2.yaml
python src\adas\train.py --config src\adas\configTests\configTest3.yaml
python src\adas\train.py --config src\adas\configTests\configTest4.yaml
python src\adas\train.py --config src\adas\configTests\configTest5.yaml