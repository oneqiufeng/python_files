# 使用python编程，在excel表格中第一列查询每一行的内容文字，通过bing搜索引擎搜索这个内容文字，然后将得到的结果里的第一条网页链接返回到对应文字的后一列，输出表格。

import pandas as pd
from bs4 import BeautifulSoup
import requests
import time

# 加载Excel文件
df = pd.read_excel('C:/Users/asus/Desktop/data.xlsx')

# 为每一行的内容文字创建一个搜索引擎查询
for index, row in df.iterrows():
    query = row[0]  # 假设第一列是我们要搜索的内容
    if pd.isna(query):  # 如果查询内容为空，则跳过
        continue
    
    # 使用Bing搜索引擎进行搜索
    search_url = f"https://www.bing.com/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    # 发送请求
    response = requests.get(search_url, headers=headers)
    time.sleep(1)  # 等待1秒，避免过快的请求

    # 解析返回的HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找第一条结果的链接
    first_link_element = soup.find('div', class_='b_algo')
    if first_link_element:
        first_link = first_link_element.find('a')['href']
    else:
        print(f"警告: 查询 '{query}' 没有返回有效结果。")
        first_link = "无有效结果"


    # 将链接保存到Excel表格中
    df.at[index, 'Link'] = first_link


# 输出表格到新的Excel文件
df.to_excel('C:/Users/asus/Desktop/output.xlsx', index=False)

print("完成！结果已保存到'output.xlsx'。")
