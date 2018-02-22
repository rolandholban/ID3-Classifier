def format(fname):
    # Turns a csv file with attribute list as first line
    # into format: [({attribute1 : value, attribute2 : value}, class)]

    formatted_data = []

    with open(fname, 'r') as file:
        attr_list = file.readline().split(",")
        for line in file:
            line = line.strip().split(",")
            entry = {}
            for i in range(len(attr_list) - 1):
                entry[attr_list[i]] = line[i]
            formatted_data.append((entry, line[-1]))
            
    return formatted_data