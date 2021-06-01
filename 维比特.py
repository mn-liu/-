from os import stat_result
import numpy as np
#已知
A = np.array([[0.5, 0.1, 0.4],
              [0.3, 0.5, 0.2],
              [0.2, 0.2, 0.6]])
#B代表了红白两色球在不同盒子中的概率
B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])
#pi初始盒子的状态概率分布
pi = np.array([[0.2],[0.3],[0.7]])
#初始化观测序列，八个结果
status = 8
O = ['红', '白', '红', '红', '白', '红', '白', '白']
#初始化一个数组来代替观测序列
o = np.zeros(status, np.int)
#红色球用0代替，白色球用1代替
for i in range (status):
    if O[i] == '红':
        o[i] = 0
    else:
        o[i] = 1
#O = np.array([[0],[1],[0]]) #0表示红色，1表示白，就是(红，白，红)观测序列
 
def weibite(A, B, pi, status, o):
    N = np.shape(A)[0] #盒子的个数
    #在时刻t状态为i的所有单个路径中概率的最大值为delta
    delta = np.zeros((status,N))
    #在时刻t状态为i的所有单个路径中概率最大的路径的第t-1个节点为fai
    fai = np.zeros((status,N)) 
    for t in range(status):
        if t == 0:
            delta[t] = pi.reshape((1, N))*np.array(B[:,o[t]]).reshape((1, N))
            continue
        for i in range(N):
            delta_t_i = delta[t-1] * A[:,i]* B[i, o[t]]
            delta[t,i] = max(delta_t_i)
            fai[t][i] = np.argmax(delta_t_i)
        print('delta:  ', delta)
        print('fai:  ', fai)
    states = np.zeros(status,np.int)
    for t in range (status):
        status = int(status)
        if t == status-1:
            states[t] = np.argmax(delta[t])
        else:
            states[t] = fai[t+1, states[t+1]]
    print('best result:', states)
    return 
            
if __name__=='__main__':
    weibite(A, B, pi, status, o)
