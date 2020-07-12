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


# import pandas
import get_data as gdt
import numpy as np
from matplotlib import pyplot as plt
from var import VectorAutoRegression

g_list = [0, 1, 1.285, 1.64, 1.96, 2.575, 3, 10]
# p_list = [0, 68, 80, 90, 95, 99, 99.9, 100]
p_list = [100, 95, 90, 85, 80, 70, 60, 0]
T = 1 # 未来T天内的预测结果,默认T为2
N = 1
SCALE = 1000

# 回测精度(data注意倒序)
def f_score(data, pre_data):
    pre = ((pre_data[:-1] - pre_data[1:])>0)[1:] # True是预测升
    real = (data[:-2] - data[1:-1])>0 # True是真升
    tp = np.sum(np.in1d(np.where(pre == True)[0], np.where(pre == real)[0]))
    fn = np.sum(np.in1d(np.where(pre == False)[0], np.where(pre != real)[0]))
    fp = np.sum(np.in1d(np.where(pre == True)[0], np.where(pre != real)[0]))
    recall = tp/(tp + fn) # 真正/(真正+假反)
    precision = tp/(tp + fp) # 真正/(真正+假正)
    F_score = 2 * recall * precision/(recall + precision)
    return F_score*100

# 用高斯分布3sigma原则估计预测可信度
def gaussian_p(x, data, id):
    f_data = gdt.FULL_DATA[id]
    z_score = (x-f_data["close"].mean())/f_data["close"].std()
    delta = np.array(data["close"].tolist()) - np.array(gdt.check_DATA[id]["close"].tolist())
    d = delta[0:-1] - delta[1:]
    dz_score = (d[0] - d.mean())/d.std()
    if abs(z_score) > 50:
        print(z_score)
        print("\n\t%s\n\n\t\t警告！！此次生成的模型稳定性较低，请注意！\n\n\t%s" % ("!!"*25, "!!"*25))
    else:
        for i in g_list:
            if abs(dz_score) < i:
                print("\n\t代码%s的股票: 根据高斯导数验证模型估计, 模型稳定性约为%s%%"%(id, p_list[g_list.index(i)-1]))
                break
            elif abs(dz_score) >= 10 and i == 10:
                print("\n\t%s\n\n\t\t警告！！拟合失败，此次生成的模型稳定性较低，请注意！\n\n\t%s" % ("!!"*25, "!!"*25))   

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
    model.fit(x, y, 1e-5, gdt.ini_data["training-times"], 1e-4)
    predict = model.predict(x)
    predict *= SCALE
    return predict[:, 0, 0]


# 作图器
def plot2d(x_lst, y_lst, data, name_lst=["real data", "predict"], n_sub=(11), title=None):

    """
    
    这是一个二维的画图器。
    
    Args:
        x_lst是一个包含了所有要画的取线的x矩阵的列表。如要画3条线就有3个矩阵;
        y_lst是一个包含了所有要画的取线的y矩阵的列表。如要画3条线就有3个矩阵;
        n_sub是你要多少个子图(subplot)的数组。不画子图则可忽略此参数。
        name_lst是每条曲线的名
        title是整个图的标题，默认为无
        
    Returns:
        不返回值，会绘制一个图表并显示出来
    """
    
    line_lst = []
    
    # 只有一个图的情况
    if n_sub == (11):
        plt.figure("2D")       # 创建一个名字叫2D的图表
        for x, y, name in zip(x_lst, y_lst, name_lst):
            line, = plt.plot(x, y, ':', label = name, marker="o")
            line_lst.append(line)
        plt.legend(line_lst, name_lst, bbox_to_anchor=(0.3, 1), loc='lower right' )
    
        
    # 有两个以上子图
    else:
        xp, yp = [int(i) for i in str(n_sub)]
        for i in range(1, xp*yp+1):
            plt.subplot(int(str(n_sub)+str(i)))
            line, = plt.plot(x_lst[i-1], y_lst[i-1], label = name_lst[i-1])
            line_lst.append(line)
        plt.legend(line_lst, name_lst, bbox_to_anchor=(0.3, -0.35), loc='lower right' )
    
    # 显示网格线(alpha是透明度)
    plt.grid(linestyle="--", alpha=0.5)
    
    # 添加标题
    if title != None:
        plt.title(title)
    
    # 修改x、y轴的刻度
    xlabels = ["-".join(i.split("-")[1:]) for i in gdt.DATA[gdt.ini_data["stocks"][data]].index.values[::-1]]
    plt.xticks(x_lst[0], xlabels, rotation=-75) # 加入字符刻度
    # plt.yticks(range(2000, 8000, 500))     # 设置y轴范围
    
    
    
    # 显示图层
    plt.show()
