import matplotlib.pyplot as plt
import networkx as nx
from time import time

G = nx.Graph()

f = open("Persephona_distance_table.txt",'r',encoding='UTF-8')

n = int(f.readline())

namelist = []
d_table = [[0.0]*n for i in range(n)]

for i in range(n):
    namelist.append (f.readline())#종 이름 읽기

f.readline()
for i in range(n):
    f.read(2)
    s = f.readline()
    s = s.split()

    for j in range(n):
        d_table[i][j] = float(s[j])#종간 유전자 거리
f.close()
#여기까지가 파일 읽기



INF = 99999
threshold = 0.144 #적당한 값 찾아야함 0.144가 적절
"""
for i in range(n):
    for j in range(n):
        if(d_table[i][j] > threshold or d_table[i][j] == -1):
            d_table[i][j] = INF
        
#거리매트릭스를 임계값을 기준으로 거리그래프로 변환
"""

for i in range(n):
    for j in range(n):
        if(i> j):
            d_table[j][i] = d_table[i][j]
#하삼각행렬을 기준으로 대칭


"""
for i in range(n):
    for j in range(n):
        if d_table[i][j] <= threshold:
            print (d_table[i][j],end = ' ')
        else:
            print (-1.00 ,end = ' ')
    print('\n')
#시험출력
"""

def dis(i,C):#유전자 i와 클러스터 C간의 거리
    avg = 0
    for j in C:
        if i == j:
            continue
        avg += d_table[i][j]
    avg /= len(C)
    return avg

def findmaxdegree(L):#리스트L의 유전자중 가장 차수가 큰 정점 찾기
    md = 0
    mi = 0
    for i in L:
        tmp = 0
        for j in range(n):
            if d_table[i][j] <= threshold:
                tmp += 1
        if md < tmp:
            md = tmp
            mi = i
    return mi


def findclose(C,S):
    ci = -1
    cd = INF
    for i in S:
        if dis(i,C) < threshold:
            if dis(i,C) < cd:
                ci = i
                cd = dis(i,C)
    return ci
#클러스터 밖의 정점중 가장가까운 정점을 찾음


def finddistant(C):
    fi = -1
    fd = 0
    for i in C:
        if dis(i,C) > threshold:
            if dis(i,C) > fd:
                fd = dis(i,C)
                fi = i
    return fi
#클러스터 안의 정점중 가장 먼 정점을 찾음 


st = time()
S = []
for i in range(n):
    S.append(i)
Clusters = []

while(len(S)):
    C = []
    mi = findmaxdegree(S)
    C.append(mi)
    S.remove(mi)
    while(True):
        ci = findclose(C,S)
        if ci != -1:
            S.remove(ci)
            C.append(ci)
        fi = finddistant(C)
        if fi != -1:
            C.remove(fi)
            S.append(fi)
        if ci == -1 and fi == -1:
            break
    C.sort()
    for i in range(len(C)):
        C[i] += 1
    Clusters.append(C)
et = time()
print(Clusters)

"""
for i in range(n):
    G.add_node(i+1)

for i in range(n):
    for j in range(n):
        if(d_table[i][j] <= threshold and i!= j):
            G.add_edge(i +1,j+1)
"""
for L in Clusters:
    for i in L:
        for j in L:
            if i != j:
                G.add_edge(i,j)

nx.draw_networkx(G,nx.circular_layout(G))
plt.show()







        
