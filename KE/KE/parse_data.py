from io import FileIO
import os 

def parse_data():
    name = []
    mass = []
    if os.path.isfile("data.txt"):

        with open(r"data.txt", "r") as d:
            c = 1
            for row in d:
                if "#" not in row:

                    row = row.split(",")
                    try:
                        n = row[0].rstrip()
                        m = float(row[1].rstrip())

                        name.append(n)
                        mass.append(m)
                    except:
                        raise TypeError("Invalid input in data.txt file on line {}".format(c))
                c +=1 

        return name,mass
    else:
        os.system("touch data.txt")
        os.system('echo # Name of Rocket Section, Mass (kg) >> data.txt')
        os.system('echo Nose cone, 1 >> data.txt')
        raise FileNotFoundError("data.txt file not found. One was created for you to fill in appropriately")
    

name,mass = parse_data()

print(name)
print(mass)