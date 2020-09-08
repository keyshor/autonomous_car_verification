#!/usr/bin/python

import continuous
import explicit_lidar

if __name__ == '__main__':
    import sys
    import csv
    with open(sys.argv[1], newline='') as f:
        for row in csv.reader(f):
            x = float(row[0])
            y = float(row[1])
            theta = float(row[2])
            explicit_err = explicit_lidar.err(x, y, theta)
            cont_err, uncertainty = continuous.err(x, y, theta)
            if explicit_err < cont_err - uncertainty or \
                    explicit_err > cont_err + uncertainty:
                print(f'explicit error {explicit_err} is outside of interval [{cont_err - uncertainty}, {cont_err + uncertainty}]')
