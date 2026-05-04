import sys
from . import convert

text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
sys.stdout.write(convert(text) + "\n")
