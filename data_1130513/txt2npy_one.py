import os
print(os.listdir("../"))
import numpy as np
import random
from sklearn.model_selection import train_test_split
def traverse_folders(folder_path):
    X=[]
    y=[]
    for files in os.listdir(folder_path):
        root=os.path.join(folder_path, files)
        if "右" in root or "R"  in root:
            if "rawdata" in root:
                print(f"正在读取文件: {root}")
                with open(file=root,mode="r",encoding="utf-8") as f:
                    # 在这里添加读取文件的代码
                    f.readline()
                    for line in f:
                        line = line.strip()
                        if line:
                            X.append(float(line))
                            y.append(float(0))

    return X,y


if __name__ == "__main__":
    folder_path = '../data_1130513/control/C-1'
    X,y=traverse_folders(folder_path)
    zipped = list(zip(X, y))
    # 打乱顺序
    random.shuffle(zipped)
    X,y=zip(*zipped)
    X=np.array(X)
    y=np.array(y)
    print(X.shape,y.shape)
    X=X.reshape(5,300,100)
    y=y.reshape(5,-1)
    X_train, X_test,y_train, y_test= train_test_split(X, y,test_size=0.2, shuffle=True, random_state=0)

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    #保存数据集，npy格式
    np.save("./X_train.npy",X_train)
    np.save("./y_train.npy", y_train)
    np.save("./X_test.npy", X_test)
    np.save("./y_test.npy", y_test)