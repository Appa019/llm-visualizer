"""Utilitários matemáticos para cálculos de redes neurais e visualização."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


def softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Calcula a função softmax de forma numericamente estável.

    Args:
        x: Array de entrada.
        axis: Eixo ao longo do qual o softmax é calculado.

    Returns:
        Array com valores softmax que somam 1 ao longo do eixo.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def mascarar_padding(scores: FloatArray, num_tokens_reais: int) -> FloatArray:
    """Retorna uma cópia dos scores com as keys de padding em -inf.

    Sem essa máscara o softmax distribui parte da atenção para posições de
    padding, e as linhas restritas aos tokens reais deixam de somar 1.

    Args:
        scores: Matriz de scores (queries, keys).
        num_tokens_reais: Quantidade de keys reais no início da sequência (>= 1).

    Returns:
        Matriz com as colunas a partir de num_tokens_reais em -inf.
    """
    if num_tokens_reais < 1:
        raise ValueError("É preciso ao menos um token real para a atenção")
    mascarado = scores.copy()
    mascarado[..., num_tokens_reais:] = -np.inf
    return mascarado


def cosine_similarity(a: FloatArray, b: FloatArray) -> float:
    """Calcula a similaridade de cosseno entre dois vetores.

    Args:
        a: Primeiro vetor.
        b: Segundo vetor.

    Returns:
        Valor de similaridade entre -1 e 1.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def entropy(probs: FloatArray) -> float:
    """Calcula a entropia de uma distribuição de probabilidade.

    Args:
        probs: Array de probabilidades que deve somar 1.

    Returns:
        Valor de entropia em nats.
    """
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def reduce_pca(
    embeddings: FloatArray,
    n_components: int = 3,
) -> list[list[float]]:
    """Reduz dimensionalidade usando PCA.

    Args:
        embeddings: Matriz de embeddings (n_amostras, n_dimensoes).
        n_components: Número de componentes principais desejado.

    Returns:
        Lista de coordenadas reduzidas como listas Python.
    """
    n_samples, n_features = embeddings.shape
    effective_components = min(n_components, n_samples, n_features)

    pca = PCA(n_components=effective_components)
    reduced = pca.fit_transform(embeddings)

    # Preenche com zeros se necessário para garantir n_components colunas
    if effective_components < n_components:
        padding = np.zeros((n_samples, n_components - effective_components))
        reduced = np.concatenate([reduced, padding], axis=1)

    return reduced.tolist()


def reduce_tsne(
    embeddings: FloatArray,
    n_components: int = 3,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> list[list[float]]:
    """Reduz dimensionalidade usando t-SNE.

    Args:
        embeddings: Matriz de embeddings (n_amostras, n_dimensoes).
        n_components: Número de dimensões de saída.
        perplexity: Parâmetro de perplexidade do t-SNE.
        random_state: Semente para reprodutibilidade.

    Returns:
        Lista de coordenadas reduzidas como listas Python.
    """
    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n_samples - 1))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=300,
    )
    reduced = tsne.fit_transform(embeddings)
    return reduced.tolist()


def reduce_dimensions(
    embeddings: FloatArray,
    method: Literal["pca", "tsne"] = "pca",
    n_components: int = 3,
) -> list[list[float]]:
    """Reduz dimensionalidade usando o método especificado.

    Args:
        embeddings: Matriz de embeddings.
        method: Método de redução ('pca' ou 'tsne').
        n_components: Número de dimensões de saída.

    Returns:
        Lista de coordenadas reduzidas.
    """
    if method == "tsne":
        return reduce_tsne(embeddings, n_components=n_components)
    return reduce_pca(embeddings, n_components=n_components)
