"""Mecanismo de Multi-Head Attention para demonstração educacional."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils.math_utils import mascarar_padding, softmax

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass
class HeadAttentionResult:
    """Resultado de atenção de uma única cabeça."""

    indice_cabeca: int
    pesos_atencao: list[list[float]]
    saida: list[list[float]]


@dataclass
class MultiHeadAttentionResult:
    """Resultado completo do mecanismo de multi-head attention."""

    cabecas: list[HeadAttentionResult]
    saida_final: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    d_model: int
    num_cabecas: int
    d_k: int


class MultiHeadAttention:
    """Implementa multi-head attention para fins educacionais.

    Cada cabeça opera em um subespaço de dimensão d_k = d_model // num_heads,
    capturando diferentes tipos de relações entre tokens.

    Args:
        d_model: Dimensão total do modelo.
        num_heads: Número de cabeças de atenção.
        seq_length: Comprimento máximo da sequência.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 8,
        seq_length: int = 10,
    ) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) deve ser divisível por num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_length = seq_length

        rng = np.random.default_rng(42)
        self.W_q_heads: list[FloatArray] = [
            rng.standard_normal((d_model, self.d_k)) * 0.1
            for _ in range(num_heads)
        ]
        self.W_k_heads: list[FloatArray] = [
            rng.standard_normal((d_model, self.d_k)) * 0.1
            for _ in range(num_heads)
        ]
        self.W_v_heads: list[FloatArray] = [
            rng.standard_normal((d_model, self.d_k)) * 0.1
            for _ in range(num_heads)
        ]
        concat_dim = num_heads * self.d_k
        self.W_o: FloatArray = rng.standard_normal((concat_dim, d_model)) * 0.1

    def _single_head_attention(
        self,
        X: FloatArray,
        W_q: FloatArray,
        W_k: FloatArray,
        W_v: FloatArray,
        num_tokens_reais: int,
    ) -> tuple[FloatArray, FloatArray]:
        """Computa atenção para uma única cabeça.

        Args:
            X: Embeddings de entrada (seq_length, d_model).
            W_q: Matriz de projeção de queries.
            W_k: Matriz de projeção de keys.
            W_v: Matriz de projeção de values.
            num_tokens_reais: Tokens reais; as keys de padding recebem peso zero.

        Returns:
            Tupla (saida, pesos_atencao).
        """
        Q: FloatArray = X @ W_q
        K: FloatArray = X @ W_k
        V: FloatArray = X @ W_v

        scores: FloatArray = Q @ K.T / np.sqrt(self.d_k)
        pesos = softmax(mascarar_padding(scores, num_tokens_reais), axis=-1)
        saida: FloatArray = pesos @ V
        return saida, pesos

    def compute_multi_head_attention(
        self,
        X: FloatArray,
        tokens: list[str],
        num_tokens_reais: int | None = None,
    ) -> MultiHeadAttentionResult:
        """Computa multi-head attention e retorna resultados por cabeça.

        Args:
            X: Embeddings de entrada (seq_length, d_model).
            tokens: Lista de tokens (incluindo padding).
            num_tokens_reais: Número de tokens reais sem padding.

        Returns:
            MultiHeadAttentionResult com dados de todas as cabeças.
        """
        if num_tokens_reais is None:
            num_tokens_reais = len([t for t in tokens if t])

        cabecas: list[HeadAttentionResult] = []
        saidas_cabecas: list[FloatArray] = []

        for i in range(self.num_heads):
            saida_cabeca, pesos = self._single_head_attention(
                X,
                self.W_q_heads[i],
                self.W_k_heads[i],
                self.W_v_heads[i],
                num_tokens_reais,
            )
            saidas_cabecas.append(saida_cabeca)
            cabecas.append(
                HeadAttentionResult(
                    indice_cabeca=i,
                    pesos_atencao=pesos.tolist(),
                    saida=saida_cabeca.tolist(),
                )
            )

        concatenado: FloatArray = np.concatenate(saidas_cabecas, axis=-1)

        # Garante compatibilidade dimensional de W_o
        if self.W_o.shape[0] != concatenado.shape[-1]:
            logger.warning(
                "Dimensao inesperada na concatenacao. Recriando W_o.",
                extra={
                    "esperado": self.W_o.shape[0],
                    "recebido": concatenado.shape[-1],
                },
            )
            rng = np.random.default_rng(42)
            self.W_o = rng.standard_normal(
                (concatenado.shape[-1], self.d_model)
            ) * 0.1

        saida_final: FloatArray = concatenado @ self.W_o

        return MultiHeadAttentionResult(
            cabecas=cabecas,
            saida_final=saida_final.tolist(),
            tokens=tokens,
            num_tokens_reais=num_tokens_reais,
            d_model=self.d_model,
            num_cabecas=self.num_heads,
            d_k=self.d_k,
        )

    def get_head_patterns(
        self,
        X: FloatArray,
        tokens: list[str],
        num_tokens_reais: int | None = None,
    ) -> list[dict[str, object]]:
        """Retorna padrões de atenção por cabeça apenas para tokens reais.

        Args:
            X: Embeddings de entrada.
            tokens: Lista de tokens.
            num_tokens_reais: Número de tokens sem padding.

        Returns:
            Lista de dicionários com padrão de atenção por cabeça.
        """
        resultado = self.compute_multi_head_attention(X, tokens, num_tokens_reais)
        n = resultado.num_tokens_reais
        tokens_reais = tokens[:n]

        return [
            {
                "cabeca": cabeca.indice_cabeca + 1,
                "tokens": tokens_reais,
                "pesos_atencao": [
                    row[:n] for row in cabeca.pesos_atencao[:n]
                ],
            }
            for cabeca in resultado.cabecas
        ]
