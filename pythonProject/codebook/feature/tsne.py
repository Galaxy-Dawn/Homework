import argparse
import sys
sys.path.append('/root/neural_decoding')  # warning!!! this should be your own root directory!!!
import torch
from codebook.model import Embedding
import os
import time
import warnings
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from pretraining.config import signal_root, feature_root, feature_codebook_number, feature_codebook_size, feature_target_length, checkpoint_root
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    print('This progress began at: ' + time.asctime(time.localtime(time.time())))
    torch.set_default_dtype(torch.float32)
    num_threads = '32'
    torch.set_num_threads(int(num_threads))
    os.environ['OMP_NUM_THREADS'] = num_threads
    os.environ['OPENBLAS_NUM_THREADS'] = num_threads
    os.environ['MKL_NUM_THREADS'] = num_threads
    os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
    os.environ['NUMEXPR_NUM_THREADS'] = num_threads

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=2)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--vocab_sizes", nargs='+', type=int)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--feature_list", nargs='+', type=str)

    parser.add_argument("--load_step", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2**18)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--decay_steps", type=int, default=500)
    parser.add_argument("--save_step", type=int, default=50)

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # for feature in args.feature_list:
    #     args.vocab_sizes += feature_codebook_size[feature]
    suffix = 'pht'
    args.feature_list = ['psd', 'hjorth', 'trend']  # 'psd', 'hjorth', 'trend'
    args.vocab_sizes = []
    for feature in args.feature_list:
        args.vocab_sizes += feature_codebook_number[feature] * [feature_codebook_size[feature]]

    checkpoint_path = '/data/share/storage/checkpoint/codebook/256_128/codebook_150_pht.pt'
    codebook = Embedding(args.vocab_sizes, args.embedding_dim, args.d_model).to(args.gpu_id)
    codebook_ckpt = torch.load(checkpoint_path, map_location=f'cuda:{args.gpu_id}', weights_only=True)
    codebook.load_state_dict(codebook_ckpt)

    codebook_embeddings = codebook.embeddings.weight.data.cpu().numpy()
    total_length = codebook_embeddings.shape[0]
    print(f'total length: {codebook_embeddings.shape}')
    part_length = total_length // 11
    parts = []
    for i in range(11):
        start_idx = i * part_length
        end_idx = (i + 1) * part_length
        parts.append(codebook_embeddings[start_idx:end_idx])


    def tsne_plot(data):
        """
        对 NumPy 数组进行 t-SNE 降维并使用 Plotly 绘制交互式散点图，
        每个点按照连续的红色 colormap 由浅到深着色。

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

        # 创建颜色映射
        colors = np.linspace(0, 1, len(data))

        # 使用 Plotly 创建交互式散点图
        fig = go.Figure(data=go.Scatter(
            x=data_2d[:, 0],
            y=data_2d[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                colorscale='Reds',  # 使用 'Reds' colormap
                size=5,
                colorbar=dict(
                    title='Sample Index',
                    tickmode='array',
                    tickvals=np.linspace(0, 1, 11),  # 11 个刻度
                    ticktext=[str(i) for i in range(0, len(data), int(len(data) / 10))]
                )
            ),
            hovertext=[f'Sample {i}' for i in range(len(data))],
            hoverinfo='text'
        ))

        # 设置图表标题和轴标签
        fig.update_layout(
            title='t-SNE Plot',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor='rgba(0,0,0,0)',  # 背景透明
            plot_bgcolor='rgba(0,0,0,0)',  # 背景透明
            xaxis=dict(showgrid=False),  # 不显示网格线
            yaxis=dict(showgrid=False)  # 不显示网格线
        )

        # 显示图表
        fig.show()


    def tsne_3d_plot(data):
        # 确保输入数据是 NumPy 数组
        if not isinstance(data, np.ndarray):
            raise ValueError("输入数据必须是 NumPy 数组")

        # 进行 t-SNE 降维
        tsne = TSNE(n_components=3, random_state=42)
        data_3d = tsne.fit_transform(data)

        # 创建颜色映射
        colors = np.linspace(0, 1, len(data))

        # 使用 Plotly 创建3D散点图
        fig = go.Figure(data=go.Scatter3d(
            x=data_3d[:, 0],
            y=data_3d[:, 1],
            z=data_3d[:, 2],
            mode='markers',
            marker=dict(
                color=colors,
                colorscale='Reds',  # 使用 'Reds' colormap
                size=5,
                colorbar=dict(title='Sample Index')
            ),
            hovertext=[f'Sample {i}' for i in range(len(data))],
            hoverinfo='text'
        ))

        # 设置图表标题和轴标签
        fig.update_layout(
            title='3D t-SNE Plot',
            scene=dict(
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2',
                zaxis_title='t-SNE Component 3',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False)
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor='rgba(0,0,0,0)',  # 背景透明
            plot_bgcolor='rgba(0,0,0,0)'  # 背景透明
        )

        # 显示图表
        fig.show()

    for i in range(11):
        tsne_plot(parts[i])
        tsne_3d_plot(parts[i])


