"""
	###########################################################################
	#		                                                                  #
	#		Project: stock_predict                                            #
	#		                                                                  #
	#		Filename: happy_birthday.py                                       #
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


import strategy as stg
import get_data as gdt
from sys import argv
import numpy as np

STOCKS = gdt.ini_data["stocks"]

# ESPM：指数平滑预测法
# VAR: 向量自回归模型
# TAYLOR: 泰勒多项式拟合

for data in range(len(gdt.DATA)):
    try:
        current_data = gdt.DATA[STOCKS[data]]
        print("\n%s\n"%(gdt.ini_data["kernel"].center(80, "~")))
        stg.alarm(current_data)
        if gdt.ini_data["kernel"] == "ESPM":
            result = [stg.tri_exp(current_data)]
            if result[-1] > current_data["close"].tolist()[0]:
                R = "接下来股市应该会升"
            elif result[-1] == current_data["close"].tolist()[0]:
                R = "接下来股市应该会保持稳定"
            else:
                R = "接下来股市可能会跌"
        else:
            if gdt.ini_data["kernel"] == "VAR":
                result = stg.var(argv, STOCKS[data])
            elif gdt.ini_data["kernel"] == "TAYLOR":
                Z = gdt.FULL_DATA[STOCKS[data]]["close"].tolist()[::-1]
                z1 = np.polyfit(Z[:-3], Z[1:-2], 3)
                p = np.poly1d(z1)
                result = p(current_data["close"][::-1])
            if result[-1] > result[-2]:
                R = "接下来股市应该会升"
            elif result[-1] == result[-2]:
                R = "接下来股市应该会保持稳定"
            else:
                R = "接下来股市可能会跌"
        # 打印结果
        print("\t%s\n\n\t\t代码为%s的股票接下来的趋势预测结果是：%.3f\n\t\t%s\n\n\t%s"%("**"*20, STOCKS[data], result[-1], R, "**"*20))
        stg.gaussian_p(result[-1], current_data, STOCKS[data])
        if gdt.ini_data["kernel"] != "ESPM":
            F_score = stg.f_score(np.array(current_data["close"].tolist()), np.array(result[::-1][:len(current_data["close"])]))
            print("\n\t回测模型, F-score为：%.2f%%"%(F_score))
        print("\n\t\t关闭图片窗口进入下一个预测\n\n%s\n"%("~~"*40))
    except:
        print("\n\t%s\n\n\t\t找不到该股票！\n\n\t%s" % ("!!"*20, "!!"*20))
    
    # 画图
    length = len(current_data["close"])
    if gdt.ini_data["kernel"] == "ESPM" or gdt.ini_data["kernel"] == "VAR":
        x = [np.arange(length)]
        y = [np.array(current_data["close"].tolist()[::-1])]
    else:
        x = [np.arange(length), np.arange(1, length+1)]
        y = [np.array(current_data["close"].tolist()[::-1]), result[:length]]
    stg.plot2d(x, y, data, n_sub=(11), title=STOCKS[data])
    
    # input("按下任意键继续~~")
