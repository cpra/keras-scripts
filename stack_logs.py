
'''
Concatenate multiple logs written by `train.py`.
~ Christopher
'''

import re
import os
import argparse

# parse args

parser = argparse.ArgumentParser(description='Concatenate log files written by `train.py`.')
parser.add_argument('--dest', type=str, required=True, help='Destination filename')
parser.add_argument('logs', type=str, nargs='+', help='Logs to concatenate in the specified order')
args = parser.parse_args()

assert(not os.path.exists(args.dest))
assert(all([os.path.isfile(f) for f in args.logs]))
assert(len(args.logs) > 1)

matcher = re.compile('^([t|e|v]) (\d+) (.+)')

with open(args.dest, 'a') as log:
    # copy first log and get epoch count
    print('{} ...'.format(args.logs[0]))

    with open(args.logs[0], 'r') as f:
        last_line = None
        for line in f:
            if line:
                log.write(line)
                last_line = line

    m = matcher.match(last_line)
    assert(m)

    epoch = int(m.group(2)) + 1
    print(' Next epoch: {}'.format(epoch))

    # concat the rest
    for fname in args.logs[1:]:
        print('{} ...'.format(fname))

        with open(fname, 'r') as f:
            for l, line in enumerate(f):
                if l > 0 and line:
                    m = matcher.match(line.strip())
                    assert(m)

                    log.write('{} {} {}\n'.format(m.group(1), epoch + int(m.group(2)), m.group(3)))

            epoch += int(m.group(2)) + 1
            print(' Next epoch: {}'.format(epoch))
