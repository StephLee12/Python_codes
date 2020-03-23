import re

regex = re.compile(r"(\w+) World")

result = regex.search("Hello World is the easiest")

if result:
    print(result.start(),result.end())

for result in regex.findall("Hello World, Bonjour World"):
    print(result)

print(regex.sub(r"\1 Earth", "Hello World"))