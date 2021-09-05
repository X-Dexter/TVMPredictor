def read_data(file_path) ->tuple:
    tmp = []

    with open(file_path,"r") as f:
        line = f.readline()
        while line is not None and len(line)>0 :
            tmp.append(line.split(","))
            line=f.readline()

    xs=[]
    ys=[]

    for data in tmp:
        xs.append(int(data[0]))
        ys.append(float(data[1]))
    return (tuple(xs),tuple(ys))