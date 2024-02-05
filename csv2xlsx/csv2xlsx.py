import pandas as pd

url= [文件路径1,文件路径2]
filename= [name1,name2]
for _ in url:
    filecsv= pd.read_csv(_)
    name= filename[url.index(_)]
    filecsv.to_excel(str(name) + '.xlsx',encoding= 'gbk')