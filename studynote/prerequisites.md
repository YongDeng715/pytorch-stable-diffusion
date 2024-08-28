# Diffusion Model

Stable Diffusion 是一个文生图(text-to-image)的深度学习模型，其基于扩散模型(diffusion model)，于2022年提出。 https://github.com/Stability-AI/stablediffusion.

## DDPM 详解

Denoising Diffusion Probabilistic Models(DDPM)
paper arxiv: https://arxiv.org/abs/2006.11239

> 马尔可夫链为状态空间中经过从一个状态到另一个状态的转换的随机过程。该过程要求具备“无记忆”的性质：**下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关**。

1. 正向加噪过程(Forward process)：$x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T$  
    前向过程中图像 $x_t$ 只与上一时刻的 $x_{t-1}$ 有关，马尔可夫过程满足：

    $$ q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})\\
    q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

2. 反向去噪过程(Reverse process)：$x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0$  
    逆向过程是去噪的过程，DDPM通过神经网络 $p_\theta(x_{t-1}|x_t)$ 来拟合 $q(x_{t-1}|x_t)$，下面式子分别为反向过程的联合概率密度和转移概率分布函数,

    $$ p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^Tp_\theta(x_{t-1}|x_t)\\
    p_\theta(x_{t-1}|x_t) = \cal N(x_{t-1}; \mu_\theta(x_t,t), \sum_\theta(x_t, t))$$

    扩散模型是隐变量模型: $p_\theta(x_0):=\int p_\theta(x_{0:T})dx_{1:T}$, 其中 $x_1, x_2, \cdots, x_{T}$是维度相同的隐变量，原始图像 $x_0 \sim q(x_0)$， 经过不断加噪的马尔可夫过程后得到纯噪声图像 $x_T \sim \mathcal{N} (x_T; 0, I)$.

## Latent Diffusion Model(LDM)
