"""
	###########################################################################
	#		                                                                  #
	#		Project: stock_predict                                            #
	#		                                                                  #
	#		Filename: strategy.py                                             #
	#		                                                                  #
	#		Programmer: Vincent Holmes (Vincent_HolmesZ@outlook.com)          #
	#		                                                                  #
	#		Description: 一个根据量化交易系统策略改造的短期股市趋势预测系统   #
	#                     ,内核与主功能分开设计,方便添加自定义内核.           #
	#                     (A short-term stock market                          #
	#                      trend prediction system                            #
	#                      transformed according t                            #
	#                     o the quantitative tradi                            #
	#                     ng system strategy, the                             #
	#                     core is designed separat                            #
	#                     ely from the main functi                            #
	#                     on, which is convenient                             #
	#                     to add a custom core.)                              #
	#		                                                                  #
	#		Start_date: 2020-07-10                                            #
	#		                                                                  #
	#		Last_update: 2020-07-10                                           #
	#		                                                                  #
	###########################################################################
"""


import pandas
import get_data as gdt
import sys
import numpy as np
import codecs as cs
from var import VectorAutoRegression

g_list = [0, 1, 1.285, 1.64, 1.96, 2.575, 3, 10]
# p_list = [0, 68, 80, 90, 95, 99, 99.9, 100]
p_list = [100, 95, 90, 85, 80, 70, 60, 0]
T = 1 # 未来T天内的预测结果,默认T为2
N = 1
SCALE = 1000


# 用高斯分布3sigma原则估计预测可信度
def gaussian_p(x, data, id):
    f_data = gdt.FULL_DATA[id]
    z_score = (x-f_data["close"].mean())/f_data["close"].std()
    delta = np.array(data["close"].tolist()) - np.array(gdt.check_DATA[id]["close"].tolist())
    d = delta[0:-1] - delta[1:]
    # input(np.array(data["close"].tolist()))
    # input(np.array(gdt.check_DATA[id]["close"].tolist()))
    dz_score = (d[0] - d.mean())/d.std()
    # input(dz_score)
    if abs(z_score) > 10:
        print("\n\t%s\n\n\t\t警告！！拟合失败，此次预测结果准确率极低，请注意！\n\n\t%s\n\n" % ("!!"*25, "!!"*25))
    else:
        for i in g_list:
            if abs(dz_score) < i:
                print("\n\t代码%s的股票: 根据高斯导数验证模型估计, 预测准确率约为%s%%\n\n\n"%(id, p_list[g_list.index(i)-1]))
                break
            elif abs(dz_score) >= 10 and i == 10:
                print("\n\t%s\n\n\t\t警告！！拟合失败，此次预测结果准确率极低，请注意！\n\n\t%s" % ("!!"*25, "!!"*25))   


# 一次指数平滑预测法
def first_exp(data):
    if gdt.ini_data["EXPMA"] == 0:
        EXPMA = data["close"][1:].mean()
    else:
        EXPMA = gdt.ini_data["EXPMA"]
    # 一次指数平滑预测结果
    result = ALPHA * data["close"][0] + (1 - ALPHA) * EXPMA
    
    with open("./ini.json", "w+", encoding="utf-8") as j:
        gdt.ini_data["EXPMA"] = result
        j.write(str(gdt.ini_data).replace("'", '"'))
    
    return result


# 二次指数平滑预测法(适用于斜率大且陡的函数)
def sec_exp(data):
    global S1
    S1 = first_exp(data)
    if gdt.ini_data["sec_EXPMA"] == 0:
        EXPMA = S1
    else:
        EXPMA = gdt.ini_data["sec_EXPMA"]
    
    S2 = ALPHA * S1 + (1 - ALPHA) * EXPMA
    at = 2*S1 - S2
    bt = at/(1 - at) * (S1 - S2)
    result = at + bt*T
    
    with open("./ini.json", "w+", encoding="utf-8") as j:
        gdt.ini_data["sec_EXPMA"] = S2
        j.write(str(gdt.ini_data).replace("'", '"'))
    
    return result


# 三次指数平滑预测法(适用于抛物线函数)
def tri_exp(data):
    S2 = sec_exp(data)
    if gdt.ini_data["tri_EXPMA"] == 0:
        EXPMA = S2
    else:
        EXPMA = gdt.ini_data["tri_EXPMA"]
    
    S3 = ALPHA * S2 + (1-ALPHA)* EXPMA
    at = 3*S1 - 3*S2 + S3
    bt = ALPHA/(2*(1 - ALPHA)**2) * ((6-5*ALPHA)*S1 - 2*(5-4*ALPHA)*S2 + (4-3*ALPHA)*S3)
    ct = ALPHA**2/(2*(1-ALPHA)**2) * (S1 - 2*S2 + S3)
    
    result = at + bt*T + T**2
    with open("./ini.json", "w+", encoding="utf-8") as j:
        gdt.ini_data["tri_EXPMA"] = S3
        j.write(str(gdt.ini_data).replace("'", '"'))
    return result


# 查询到的数据为单只股票而非基金指数时提醒
def alarm(data):
    if data["open"][2]<1000:
        print("\n\n\t%s\n\n\t\t警告！此次查询结果为个股数据！！\n\n\t%s\n" % ("!!"*25, "!!"*25))
        
ALPHA = 2/ (gdt.ini_data["day_back"] + 1)

# 生成训练数据
def make_training_date(data, log_phaseP=1):
    x = [[] for p in range(log_phaseP)]
    y = []
    for i in range(len(data)):
        y.append(data[i])
        for j in range(log_phaseP):
            offset = i - (j + 1)
            if 0 <= offset and offset < len(data):
                x[j].append(data[offset])
            else:
                x[j].append(0.0)
    return x, y

# VAR模型生成
def var(args, id):
    log_phaseP = int(args[1]) if len(args) >= 2 else 1
    data = gdt.FULL_DATA[id]
    data = data["close"].tolist()
    x, y = make_training_date(data, log_phaseP)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape((len(data), N, 1))
    x = x.reshape((log_phaseP, len(data), N, 1))
    x = x / SCALE  # scale
    y = y / SCALE  # scale
    model = VectorAutoRegression(n=N, P=log_phaseP)
    model.fit(x, y, 1e-5, 1000, 1e-4)
    predict = model.predict(x)
    predict *= SCALE
    return predict[-1, 0, 0]