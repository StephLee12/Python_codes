import re

regex = r"([a-zA-Z]+) (\d+)"

print(re.sub(regex,r"\2 of \1", "June 24, August 9, Dec 12"))