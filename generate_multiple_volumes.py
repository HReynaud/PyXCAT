from subprocess import Popen
import argparse
from math import log10 as log10

def leading_zeros(num, digits):
    """
    Returns a string of digits with leading zeros.
    """
    return str(num).zfill(digits)

parser = argparse.ArgumentParser(description='Generates multiple volumes in parallel.')

parser.add_argument('-c', '--count', type=int, help='Number of volumes to generate.')
parser.add_argument('-s', '--size', type=int, help='Size of the volume to generate.')
parser.add_argument('-t', '--types', type=str, help='Type of the volume to generate [CT, SEG, NURBS, ATN].')
parser.add_argument('-n', '--name', type=str, help='Base file name.')
parser.add_argument('-o', '--output', type=str, help='Path of the output file.')
parser.add_argument('-p', '--program', type=str, help='Path of the XCAT program folder.')
parser.add_argument('-f', '--frames', type=int, help='Number of frames.')

parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')

parser.add_argument('--keep_raw', action='store_true', help='Keep the raw files generated by XCAT.')
parser.add_argument('--preview', action='store_true', help='Generate a preview image sampled from the volume.')
parser.add_argument('--randomize', action='store_true', help='Generate a preview image sampled from the volume.')

args = parser.parse_args()
print(args)

commands = [
    'python generate_volume.py -n ' + args.name+str(0).zfill(int(log10(args.count)+1))
    + ' -s '+str(args.size)
    + ' -t '+args.types
    + ' -o '+args.output
    + ' -p '+args.program
    + ' -f '+str(args.frames)
    + ' -v' if args.verbose else ''
    + '--keep_raw' if args.verbose else ''
    + '--preview' if args.preview else ''
    + '--randomize'
    for i in range(args.count)]
print(commands)

procs = [ Popen(i.split(" ")) for i in commands ]
for p in procs:
   p.wait()

print("Done.")