def up_data(value):
    data = list()
    for i in value:
        if data:
            data.append(data[-1] * 0.9 + i * 0.1)
        else:
            data.append(i)
    return data


if __name__ == '__main__':
    data = up_data([10, 11, 12, 13, 14, 15, 16])
    print(data)
