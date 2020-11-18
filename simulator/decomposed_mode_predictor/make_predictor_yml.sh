#!/bin/sh

for STEM in big straight_little square_right_little square_left_little sharp_right_little sharp_left_little
do
	python ../../train_td3/h5_to_yaml.py $STEM.h5 $STEM.yml
done
