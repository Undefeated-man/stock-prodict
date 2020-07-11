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
import sys

# ESPM：指数平滑预测法
# VAR: 向量自回归模型

for data in range(len(gdt.DATA)):
    # try:
    # input(gdt.DATA)
    stg.alarm(gdt.DATA[gdt.ini_data["stocks"][data]])
    if gdt.ini_data["kernel"] == "ESPM":
        result = stg.tri_exp(gdt.DATA[gdt.ini_data["stocks"][data]])
    elif gdt.ini_data["kernel"] == "VAR":
        result = stg.var(sys.argv, gdt.ini_data["stocks"][data])
    # 打印结果
    # input(gdt.DATA)
    if result > gdt.DATA[gdt.ini_data["stocks"][data]]["close"].tolist()[0]:
        R = "接下来股市应该会升"
    elif result == gdt.DATA[gdt.ini_data["stocks"][data]]["close"].tolist()[0]:
        R = "接下来股市应该会保持稳定"
    else:
        R = "接下来股市可能会跌"
    print("\t%s\n\n\t\t代码为%s的股票接下来三天的趋势预测结果是：%.3f\n\t\t%s\n\n\t%s"%("**"*20, gdt.ini_data["stocks"][data], result, R, "**"*20))
    stg.gaussian_p(result, gdt.DATA[gdt.ini_data["stocks"][data]], gdt.ini_data["stocks"][data])
    # except:
    #     print("\n\t%s\n\n\t\t找不到该股票！\n\n\t%s" % ("!!"*20, "!!"*20))

input()