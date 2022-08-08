import ccxt 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
import FinanceDataReader as fdr
class download_data:
  def __init__(self,since='2018-01-01 00:00:00',code_name = 'BTC/USDT',frame = '1d',limit=10,data_amount = 1000):
    self.since = since
    self.code_name = code_name
    self.frame = frame
    self.limit = limit
    self.data_amount = data_amount
  def to_timestemp(self,dt):
    dt = datetime.datetime.timestamp(dt)
    dt = int(dt) * 1000
    return dt
  def set_df(self,dt):
    binance = ccxt.binance()
    col = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    btc_ohlcv = binance.fetch_ohlcv(self.code_name,limit = self.data_amount,timeframe=self.frame,since=dt)
    df = pd.DataFrame(btc_ohlcv,columns=col)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df
  def concat_data(self):
    format = '%Y-%m-%d %H:%M:%S'
    dt = datetime.datetime.strptime(self.since,format)
    dt = self.to_timestemp(dt)
    df = self.set_df(dt)
    data = df
    for i in range(self.limit):
      dt = self.to_timestemp(df.index[-1])
      df = self.set_df(dt)
      data = pd.concat([data,df])
    return data
  def down_data(self):
    return self.concat_data()

def find_max(data,idx,term):
  for i in range(idx+1,idx+term+1):
    if data.close[i] > data.close[idx]:
      return -1
  return 1
def find_min(data,idx,term):
  for i in range(idx+1, idx+term+1):
    if data.close[i] < data.close[idx]:
      return -1
  return 1
def high_price_range(data,idx,term):
  price = data.iloc[idx]['close']
  start = idx-term
  result = []
  if start < 0:
    start = 0
  for i in range(start, idx+term+1):
    if data.iloc[i]['close'] >= price * 0.99:
       result.append(data.index[i])
  return result
def low_price_range(data,idx,term):
  price = data.iloc[idx]['close']
  result = []
  start = idx-term
  if start < 0:
    start = 0
  for i in range(start, idx+term+1):
    if data.iloc[i]['close'] <= price * 1.01:
       result.append(data.index[i])
  return result
def transpose(df,high_range,low_range):
  index = df.index
  temp = []
  for z in range(len(df)):
    cnt = 0
    # 고점
    for i in high_range:
      for j in i:
        if df.index[z] == j:
          temp.append(1)
          cnt = 1
          break
    if cnt != 0:
      continue
    # 저점
    else:
      for i in low_range:
        for j in i:
          if df.index[z] == j:
            temp.append(-1)
            cnt = 1
            break
    if cnt != 0:
      continue
    # 고점, 저점 아닌 점
    else:
      temp.append(0)
  df['label'] = temp
  return df
def previous_labeling(df,previous_range,term):

  max_index = df[df.close == max(df.close[:previous_range])].index[0]
  min_index = df[df.close == min(df.close[:previous_range])].index[0]
  max_num = [i for i in range(len(df)) if max_index == df.index[i]]
  min_num = [i for i in range(len(df)) if min_index == df.index[i]]
  start_index = 1

  if max_index > min_index:
    Last_label = max_index
    ck,start_index = 1,max_num[0]

    if find_max(df,max_num[0],term) != 1:
      Last_label = min_index
      ck,start_index = 0,min_num[0]

  elif min_index > max_index:
    Last_label = min_index
    ck,start_index = 0,min_num[0]

    if find_min(df,min_num[0],term) != 1:
      Last_label = max_index
      ck,start_index = 1,max_num[0]

  return Last_label,ck,start_index+1
def price_range(df,high_label,low_label,alpha):
  high_range = []
  low_range = []
  for i in range(len(high_label)):
    high_range.append(high_price_range(df,high_label[i][0],alpha))
  for i in range(len(low_label)):
    low_range.append(low_price_range(df,low_label[i][0],alpha))
  return high_range,low_range
def draw_label(df,delta,code):
  fig = plt.figure(figsize=(25,13))
  fig = plt.plot(df.close,color='g',alpha = 0.8)
  line = []
  for i in range(len(df)):
    line2 = []
    if df.label[i] == 1:
      fig = plt.scatter(df.index[i],df.iloc[i]['close'],color='r')
      # fig = plt.annotate(df.index[i], xy = (df.index[i+1], df.iloc[i]['close']-1000))
      line2.append(df.index[i])
      line2.append(df.iloc[i]['close'])
      line.append(line2)
    elif df.label[i] == -1:
      fig = plt.scatter(df.index[i],df.iloc[i]['close'],color='b')
      # fig = plt.annotate(df.index[i], xy = (df.index[i+1], df.iloc[i]['close']-1000))
      line2.append(df.index[i])
      line2.append(df.iloc[i]['close'])
      line.append(line2)
  
  # if len(line)%2 == 1:
  #   line.pop(len(line)-1)
  

  # 상승 : red
  # 하락 : blue
  # 횡보 : gray
  gap_mean = 0
  sum_temp = 0
  for i in range(len(line)-1):
    sum_temp += abs(line[i][1] - line[i+1][1])
  gap_mean = sum_temp / (len(line)-1)
  print(gap_mean)
  for i in range(0,len(line)-1,1):
    # 상승 구간
    if line[i][1] + (gap_mean * delta) < line[i + 1][1]:
      fig = plt.plot([line[i][0],line[i+1][0]], [line[i][1],line[i+1][1]],color='red',alpha = 0.5)
      pass
    # 하락 구간
    elif line[i][1] > line[i + 1][1] + (gap_mean * delta):
      fig = plt.plot([line[i][0],line[i+1][0]], [line[i][1],line[i+1][1]],color='blue', alpha = 0.5)
      pass
    # 횡보 구간
    else:
      fig = plt.plot([line[i][0],line[i+1][0]], [line[i][1],line[i+1][1]],color='black')
  

  fig = plt.savefig("Labeling_" + str(code) +".png")
def labeling(df,term = 20,alpha = 3,previous_range = 50,delta_w = 1.1):
  # 고점은 최고점 대비 -1% 이하
  # 저점은 최저점 대비 +1% 이상
  previous_label,ck,start_index = previous_labeling(df,previous_range,term)
  temp_line = []
  high_label = []
  low_label = []

  if ck == 0:
    temp_line.append(start_index-1)
    temp_line.append(df.index[start_index-1])
    low_label.append(temp_line)

    max = df.close[start_index-1]
    previous_min = df.close[start_index-1]
    min = 99999999
  else:
    temp_line.append(start_index-1)
    temp_line.append(df.index[start_index-1])
    high_label.append(temp_line)

    previous_max = df.close[start_index-1]
    min_index = [start_index-1]
    min = df.close[start_index-1]
    max = 0

    temp_index = 0
  for i in range(start_index,len(df)-term):
    line = []
    if ck == 0: # 고점을 찾는다
      if df.close[i] > max and df.close[i] >= previous_min * delta_w: # 만약 현재 인덱스 값이 max보다 클시
        if find_max(df,i,term) == 1: # 만약에 앞에 term일 동안 더 큰 값이 없으면 고점이라고 판단
          line.append(i)
          line.append(df.index[i])
          high_label.append(line)
          previous_max = df.close[i]
          max = 0
          ck = 1

          temp_index = i
        elif find_max(df,i,term) == -1: # 만약 앞에 term일 동안 더 큰 값이 있으면 고점이 아님 , max값만 업데이트
          max = df.close[i]
    elif ck == 1: # 저점을 찾는다.
      if df.close[i] < min and df.close[i] <= previous_max * (2-delta_w):
        if find_min(df,i,term) == 1:
          line.append(i)
          line.append(df.index[i])
          low_label.append(line)
          previous_min = df.close[i]
          min = 99999999
          ck = 0

          temp_index = i
        elif find_min(df,i,term) == -1:
          min = df.close[i]

  
  last_index = -1
  for i in range(temp_index,len(df)):
    if df.close[i] == max:
      last_index = i
      
    elif df.close[i] == min:
      last_index = i
      
    elif df.close[i] > max and ck == 0:
      last_index = i
      max = df.close[i]
      
    elif df.close[i] < min and ck == 1:
      last_index = i
      min = df.close[i]
    
  if ck == 0:
    line.append(last_index)
    line.append(df.index[last_index])
    high_label.append(line)
  else:
    line.append(last_index)
    line.append(df.index[last_index])
    low_label.append(line)
  high_range,low_range = price_range(df,high_label,low_label,alpha)
  return transpose(df,high_range,low_range)
def FinanceData(code,start,end):
    df = fdr.DataReader(str(code), start, end)
    df = df.drop(['Change'],axis = 1)
    col = [i.lower() for i in df.columns.to_list()]
    df.columns = col
    return df
class LabelingData:
  def __init__(self,code,start = '2010-01-01',end = '2022-01-01',term = 20,previous_range = 20,delta = 0.3):
    self.term = term
    self.previous_range = previous_range
    self.code = code
    self.start = start
    self.end = end
    self.delta = delta
    self.data = FinanceData(code,start,end)
    labeling_data = labeling(self.data,term = 30,alpha = 0,previous_range = 30,delta_w = 1) # lookahead range, alpha
    draw_label(labeling_data,self.delta,self.code)


# code : 종목코드
# start : 데이터 시작 일자
# end : 데이터 종료 일자
# term : 중단기 라벨링을 결정하는 파라미터, 20, 60 으로 조정
# previous_range : 처음 고점,저점 라벨 결정하는 알고리즘을 수행할 때 데이터 범위 -> 많이 중요하지 않음, 디폴트로 사용해도 크게 상관없을 듯..?
# delta : 횡보구간을 위한 파라미터, (각 고점과 저점의 갭 평균 * delta)를 통해서 횡보구간을 판단한다. delta가 작을 수록 횡보구간이 덜 잡힘, 클수록 많은 횡보구간이 잡힌다.
LabelingData(code = '131100',start = '2010-01-01',end = '2022-01-01',term = 20, previous_range = 20,delta = 0.3) # 현재 파라미터가 디폴트로 설정된 파라미터 (단, code는 디폴트 없습니다)