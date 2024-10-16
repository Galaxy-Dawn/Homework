import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(data):
    """
    对 NumPy 数组进行 t-SNE 降维并绘制散点图。
    参数:
    data (numpy.ndarray): 输入的高维数据，形状为 (n_samples, n_features)。
    返回:
    None
    """
    # 确保输入数据是 NumPy 数组
    if not isinstance(data, np.ndarray):
        raise ValueError("输入数据必须是 NumPy 数组")
    # 进行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=50, alpha=0.7)
    plt.title('t-SNE Plot')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 生成一些示例数据
    np.random.seed(42)
    data = np.random.randn(100, 10)  # 100个样本，每个样本有10个特征

    # 调用 tsne_plot 函数
    tsne_plot(data)


