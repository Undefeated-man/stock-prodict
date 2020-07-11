"""
	###########################################################################
	#		                                                                  #
	#		Project: stock_predict                                            #
	#		                                                                  #
	#		Filename: get_data.py                                             #
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


import tushare as ts # 用于查历史数据
#import easyquotation # 用于查询实时数据
import json
import time
import datetime

DATA = {}
check_DATA = {}
FULL_DATA = {}
BT = 1
AD = 0

# 生成格式化日期
def stfdate(ls):
    return "-".join([str(i) if i>9 else "0"+str(i) for i in ls])


# 计算日期
def day_range(n=0, ad=0):
    epsimon = ini_data["day_back"]//4 - 1
    ed = datetime.datetime.now() - datetime.timedelta(days=n)
    d = ed - datetime.timedelta(days=ini_data["day_back"]+2*epsimon+ad)
    return stfdate([d.year, d.month, d.day]), stfdate([d.year, ed.month, ed.day])



# 读取json配置数据(股票代码，指数平滑预测观测时间)
with open("ini.json", "r", encoding="utf-8") as j:
    ini_data = json.loads(j.read())


# 获取某只股票的全数据
for i in range(len(ini_data["stocks"])):
    FULL_DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i])
# def full_data(id):
    # return ts.get_hist_data(id)

# 获取股票数据
s_t, e_t = day_range()
for i in range(len(ini_data["stocks"])):
    DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i], start=s_t, end=e_t)
    while len(DATA[ini_data["stocks"][i]])<30:
        # input(len(DATA[ini_data["stocks"][i]]))
        AD += 1
        s_t, e_t = day_range(ad=AD)
        DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i], start=s_t, end=e_t)

# 获取验证信息
AD = 0
cs_t, ce_t = day_range(BT)
for i in range(len(ini_data["stocks"])):
    check_DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i], start=cs_t, end=ce_t)
    while check_DATA[ini_data["stocks"][i]]["close"][2] == DATA[ini_data["stocks"][i]]["close"][2]:
        BT += 1
        check_DATA[ini_data["stocks"][i]].drop(check_DATA[ini_data["stocks"][i]].index.values[0], inplace = True)
        cs_t, ce_t = day_range(BT)
        check_DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i], start=cs_t, end=ce_t)
    while len(check_DATA[ini_data["stocks"][i]])<30:
        AD += 1
        cs_t, ce_t = day_range(BT, ad=AD) 
        check_DATA[ini_data["stocks"][i]] = ts.get_hist_data(ini_data["stocks"][i], start=cs_t, end=ce_t)
    
if __name__ == "__main__":
    print(DATA)