def aminoCount(seq):
    count = {}

    for char in seq:
        count[char] = seq.count(char)
    
    return count

seq = ""

with open("G3V5D1.txt") as file:
    file.readline()
    for line in file:
        for char in line:
            if char == '\n':
                continue
            else:
                seq += char
            
print(aminoCount(seq))