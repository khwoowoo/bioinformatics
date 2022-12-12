import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import time

#데이터셋 읽어오기
df = pd.read_excel('bio dataset.xls')
data = df.loc[:, 'A':'Y']

# 정규화 진행
scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)

"""
# 엘보우 기법(오차제곱합의 값이 최소가 되도록 결정하는 방법)
inertia_arr = []
k_range = range(2, 11)

for k in k_range :
    Kmeans = KMeans(n_clusters=k, random_state=200)
    Kmeans.fit(data_scale)
    interia = Kmeans.inertia_
    inertia_arr.append(interia)
inertia_arr = np.array(inertia_arr)

plt.plot(k_range, inertia_arr)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
"""

"""
#실루엣 기법
silhouette_vals = []
for i in range(2, 11):
    kmeans_plus = KMeans(n_clusters=i, init='k-means++')
    pred = kmeans_plus.fit_predict(data_scale)
    silhouette_vals.append(np.mean(silhouette_samples(data_scale, pred, metric='euclidean')))

plt.plot(range(2, 11), silhouette_vals, marker='o')
plt.title('Silhouette method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
"""

"""
# 적절한 k 입력
k = int(input('k를 입력하세요: '))
"""

k = 3

# k means 시간측정 시작
start_time = time.time()

# 그룹 수, random_state 설정
model = KMeans(n_clusters = k, random_state = 10)

# 정규화된 데이터에 학습
model.fit(data_scale)

# k means 시간측정 종료, 측정된 시간 출력
end_time = time.time()
print(f"{end_time - start_time:.5f} sec")

# 번호 저장
df['num'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

# 클러스터링 결과 각 데이터가 몇 번째 그룹에 속하는지 저장
df['cluster'] = model.fit_predict(data_scale)

# 데이터셋의 25개 열에서 PCA를 이용하여 2차원으로 차원축소
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(data)
df['pca_x'] = pca_transformed[:, 0]
df['pca_y'] = pca_transformed[:, 1]


# 원본 Data Points 시각화
plt.figure(figsize = (8, 8))
plt.scatter(df['pca_x'], df['pca_y'], color='k')
plt.title('Original Data Points')
plt.xlabel('pca_x')
plt.ylabel('pca_y')
plt.show()


# K-means 결과 시각화
plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'pca_x'], df.loc[df['cluster'] == i, 'pca_y'],
                label = 'cluster ' + str(i))

for pca_x, pca_y, num in np.array(df[['pca_x', 'pca_y', 'num']]):
    plt.annotate(num, (pca_x+0.02, pca_y+0.06))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('pca_x', size = 12)
plt.ylabel('pca_y', size = 12)
plt.show()