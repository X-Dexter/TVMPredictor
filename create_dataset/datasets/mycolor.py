def mycolor(index)->str:
    colors = ['red','green','blue','c','m','y','k']

    if index<=len(colors) and index>0:
        return colors[index-1]
    else:
        return 'w'