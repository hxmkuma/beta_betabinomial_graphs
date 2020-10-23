import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom
import streamlit as st

# 乱数シード固定
np.random.seed(0)

# sidebar におけるパラメータを設定
st.sidebar.markdown("Set Parameter")
a = st.sidebar.number_input('parameter a', min_value=0.0, value=10.0)
b = st.sidebar.number_input('parameter b', min_value=0.0, value=100.0)
n = st.sidebar.number_input('input n', value = 50)
size = st.sidebar.number_input('sample size', value=1000)

st.header("An Interactive Graph of Beta Distribution")

# 数式表現
st.subheader('Mathematical expression')
st.text('Beta distribution is defined by:')
st.latex(r'''
f(x) = \left\{ \begin{array}{ll}
    \frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1} & (x \leq 0 \leq 1) \\
    0 & (x \leq 0 \vee 1 \leq x)
\end{array} \right.
''')
st.text('Beta Binomial distribution is defined by:')
st.latex(r'''
P(X=x) = {}_n C_r \frac{B(a+x, b+n-x)}{B(a,b)}
''')

# グラフ設定
fig = plt.figure(figsize=(18,14))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# ベータ分布のグラフ
x_beta = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), size)
x_betabinom = np.arange(betabinom.ppf(0, n, a, b), betabinom.ppf(1, n, a, b)) + 1
ax1.plot(x_beta, beta.pdf(x_beta,a,b), color="blue", label=f"Beta : a = {a}, b = {b}")
ax1.legend(loc='upper center', fontsize=30)
ax1.tick_params(labelsize=24)

# ベータ2項分布のグラフ
ax2.plot(x_betabinom, betabinom.pmf(x_betabinom,n,a,b), color="red", alpha = 0.5, label=f"Beta Binomial : a = {a}, b = {b}")
ax2.vlines(x_betabinom, 0, betabinom.pmf(x_betabinom,n,a,b), lw=16, alpha = 0.55, color="red")
ax2.legend(loc="upper center", fontsize=30)
ax2.tick_params(labelsize=24)
fig.tight_layout()

# 可視化
st.subheader("Visualization of the probability functions")
st.pyplot(fig)