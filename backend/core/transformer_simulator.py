"""Simulador de Transformer: embeddings, positional encoding e self-attention."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from utils.math_utils import mascarar_padding, softmax

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass
class EmbeddingResult:
    """Resultado do cálculo de embeddings."""

    token_embeddings: list[list[float]]
    positional_encoding: list[list[float]]
    embeddings_finais: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    padroes_sinusoidais: list[list[float]]  # primeiras 10 dims por posição


@dataclass
class SelfAttentionResult:
    """Resultado passo a passo do self-attention."""

    Q: list[list[float]]
    K: list[list[float]]
    V: list[list[float]]
    scores_escalados: list[list[float]]
    pesos_atencao: list[list[float]]
    saida: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    fator_escala: float


@dataclass
class AttentionFlowResult:
    """Fluxo de atenção para um token específico."""

    token_alvo: str
    indice_token: int
    pesos: list[float]
    tokens: list[str]
    conexoes_significativas: list[dict[str, float | str | int]]


@dataclass
class TokenImportanceResult:
    """Métricas de importância por token."""

    tokens: list[str]
    importancia_recebida: list[float]
    importancia_dada: list[float]
    importancia_combinada: list[float]


@dataclass
class AttentionPatternResult:
    """Padrões de atenção analisados."""

    tokens: list[str]
    matriz_atencao: list[list[float]]
    distribuicao: dict[str, float]
    atencao_por_token: list[dict[str, float | str]]


class TransformerSimulator:
    """Simula os componentes de um Transformer para fins educacionais.

    Gera dados determinísticos com numpy (seed fixo) para demonstrar
    embeddings, positional encoding e mecanismo de self-attention.

    Args:
        d_model: Dimensão do modelo (tamanho dos vetores de embedding).
        seq_length: Comprimento máximo da sequência.
        vocab_size: Tamanho do vocabulário simulado.
    """

    def __init__(
        self,
        d_model: int = 64,
        seq_length: int = 10,
        vocab_size: int = 1000,
    ) -> None:
        self.d_model = d_model
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        rng = np.random.default_rng(42)
        self.W_q: FloatArray = rng.standard_normal((d_model, d_model)) * 0.1
        self.W_k: FloatArray = rng.standard_normal((d_model, d_model)) * 0.1
        self.W_v: FloatArray = rng.standard_normal((d_model, d_model)) * 0.1

    # ------------------------------------------------------------------
    # Positional Encoding
    # ------------------------------------------------------------------

    def create_positional_encoding(self) -> FloatArray:
        """Cria encoding posicional sinusoidal (Vaswani et al., 2017).

        Returns:
            Matriz (seq_length, d_model) com encodings posicionais.
        """
        pe = np.zeros((self.seq_length, self.d_model))
        posicoes = np.arange(self.seq_length)[:, np.newaxis]
        dims_pares = np.arange(0, self.d_model, 2)
        divisores = np.power(10000.0, dims_pares / self.d_model)

        pe[:, 0::2] = np.sin(posicoes / divisores)
        if self.d_model % 2 == 0:
            pe[:, 1::2] = np.cos(posicoes / divisores)
        else:
            pe[:, 1::2] = np.cos(posicoes / divisores[:-1])

        return pe

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def create_embeddings_from_tokens(
        self, tokens: list[str]
    ) -> tuple[FloatArray, list[str], int]:
        """Cria embeddings simulados para os tokens fornecidos.

        Usa heurísticas de categoria gramatical para diferenciar os embeddings:
        artigos, preposições e pontuação recebem pesos menores, e conteúdos
        lexicais (substantivos, verbos) recebem um viés positivo.

        Args:
            tokens: Lista de tokens (strings).

        Returns:
            Tupla (embeddings, tokens_usados, num_tokens_reais) onde
            embeddings tem shape (seq_length, d_model).
        """
        tokens = tokens[: min(len(tokens), self.seq_length)]
        num_tokens_reais = len(tokens)

        tokens_padded = list(tokens)
        while len(tokens_padded) < self.seq_length:
            tokens_padded.append("")

        rng = np.random.default_rng(42)
        embeddings: FloatArray = rng.standard_normal(
            (self.seq_length, self.d_model)
        ) * 0.5

        artigos = {"o", "a", "os", "as", "um", "uma"}
        preposicoes = {"de", "em", "para", "com", "por", "do", "da", "no", "na"}
        pontuacao = {".", ",", "!", "?", ";", ":"}

        for i, token in enumerate(tokens_padded):
            if i >= num_tokens_reais:
                embeddings[i] *= 0.1
            elif token in artigos:
                embeddings[i] *= 0.8
            elif token in preposicoes:
                embeddings[i] *= 0.7
            elif token in pontuacao:
                embeddings[i] *= 0.5
            else:
                embeddings[i] += 0.3

        return embeddings, tokens_padded, num_tokens_reais

    def compute_embeddings(self, tokens: list[str]) -> EmbeddingResult:
        """Computa embeddings completos com positional encoding.

        Args:
            tokens: Lista de tokens de entrada.

        Returns:
            EmbeddingResult com todos os dados necessários para visualização.
        """
        embeddings, tokens_usados, num_tokens_reais = (
            self.create_embeddings_from_tokens(tokens)
        )
        pos_encoding = self.create_positional_encoding()
        embeddings_finais = embeddings + pos_encoding

        # Primeiras 10 dimensões para visualização de padrões sinusoidais
        n_dims = min(10, self.d_model)
        padroes = pos_encoding[:, :n_dims].tolist()

        return EmbeddingResult(
            token_embeddings=embeddings.tolist(),
            positional_encoding=pos_encoding.tolist(),
            embeddings_finais=embeddings_finais.tolist(),
            tokens=tokens_usados,
            num_tokens_reais=num_tokens_reais,
            padroes_sinusoidais=padroes,
        )

    # ------------------------------------------------------------------
    # Self-Attention
    # ------------------------------------------------------------------

    def compute_attention_step_by_step(
        self,
        X: FloatArray,
        tokens: list[str],
        num_tokens_reais: int,
    ) -> SelfAttentionResult:
        """Computa self-attention passo a passo, retornando todos os intermediários.

        Implementa: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

        Args:
            X: Matriz de embeddings de entrada (seq_length, d_model).
            tokens: Lista de tokens (incluindo padding).
            num_tokens_reais: Quantidade de tokens reais (sem padding).

        Returns:
            SelfAttentionResult com Q, K, V, scores, pesos e saída.
        """
        Q: FloatArray = X @ self.W_q
        K: FloatArray = X @ self.W_k
        V: FloatArray = X @ self.W_v

        fator_escala = float(np.sqrt(self.d_model))
        scores = Q @ K.T
        scores_escalados: FloatArray = scores / fator_escala
        pesos_atencao = softmax(mascarar_padding(scores_escalados, num_tokens_reais), axis=-1)
        saida: FloatArray = pesos_atencao @ V

        return SelfAttentionResult(
            Q=Q.tolist(),
            K=K.tolist(),
            V=V.tolist(),
            scores_escalados=scores_escalados.tolist(),
            pesos_atencao=pesos_atencao.tolist(),
            saida=saida.tolist(),
            tokens=tokens,
            num_tokens_reais=num_tokens_reais,
            fator_escala=fator_escala,
        )

    def compute_attention_flow(
        self,
        pesos_atencao: FloatArray,
        tokens: list[str],
        indice_token: int,
        num_tokens_reais: int | None = None,
    ) -> AttentionFlowResult:
        """Calcula o fluxo de atenção a partir de um token específico.

        Args:
            pesos_atencao: Matriz de pesos de atenção (seq_length, seq_length).
            tokens: Lista de tokens.
            indice_token: Índice do token alvo.
            num_tokens_reais: Número de tokens reais (sem padding).

        Returns:
            AttentionFlowResult com pesos e conexões significativas.
        """
        if num_tokens_reais is None:
            num_tokens_reais = len([t for t in tokens if t])

        indice_token = min(indice_token, num_tokens_reais - 1)
        pesos_token = pesos_atencao[indice_token, :num_tokens_reais]

        conexoes = [
            {
                "token": tokens[i],
                "indice": i,
                "peso": float(pesos_token[i]),
            }
            for i in range(num_tokens_reais)
            if float(pesos_token[i]) > 0.05
        ]
        conexoes.sort(key=lambda c: c["peso"], reverse=True)  # type: ignore[arg-type]

        return AttentionFlowResult(
            token_alvo=tokens[indice_token],
            indice_token=indice_token,
            pesos=pesos_token.tolist(),
            tokens=tokens[:num_tokens_reais],
            conexoes_significativas=conexoes,
        )

    def analyze_attention_patterns(
        self,
        pesos_atencao: FloatArray,
        tokens: list[str],
        num_tokens_reais: int | None = None,
    ) -> AttentionPatternResult:
        """Analisa padrões estatísticos da matriz de atenção.

        Args:
            pesos_atencao: Matriz de pesos de atenção.
            tokens: Lista de tokens.
            num_tokens_reais: Número de tokens reais.

        Returns:
            AttentionPatternResult com análise estatística.
        """
        if num_tokens_reais is None:
            num_tokens_reais = len([t for t in tokens if t])

        tokens_reais = tokens[:num_tokens_reais]
        pesos_reais: FloatArray = pesos_atencao[
            :num_tokens_reais, :num_tokens_reais
        ]

        distribuicao = {
            "media": float(pesos_reais.mean()),
            "desvio_padrao": float(pesos_reais.std()),
            "minimo": float(pesos_reais.min()),
            "maximo": float(pesos_reais.max()),
        }

        atencao_por_token = [
            {
                "token": tokens_reais[i],
                "atencao_recebida": float(pesos_reais[:, i].sum()),
                "atencao_dada": float(pesos_reais[i, :].sum()),
            }
            for i in range(num_tokens_reais)
        ]

        return AttentionPatternResult(
            tokens=tokens_reais,
            matriz_atencao=pesos_reais.tolist(),
            distribuicao=distribuicao,
            atencao_por_token=atencao_por_token,
        )

    def compare_attention_patterns(
        self,
        pesos1: FloatArray,
        tokens1: list[str],
        num_tokens1: int,
        pesos2: FloatArray,
        tokens2: list[str],
        num_tokens2: int,
    ) -> dict[str, object]:
        """Compara padrões de atenção entre duas sequências.

        Args:
            pesos1: Matriz de atenção da sequência 1.
            tokens1: Tokens da sequência 1.
            num_tokens1: Tokens reais da sequência 1.
            pesos2: Matriz de atenção da sequência 2.
            tokens2: Tokens da sequência 2.
            num_tokens2: Tokens reais da sequência 2.

        Returns:
            Dicionário com análise comparativa de ambas as sequências.
        """
        analise1 = self.analyze_attention_patterns(pesos1, tokens1, num_tokens1)
        analise2 = self.analyze_attention_patterns(pesos2, tokens2, num_tokens2)

        return {
            "frase_1": {
                "tokens": analise1.tokens,
                "matriz_atencao": analise1.matriz_atencao,
                "distribuicao": analise1.distribuicao,
                "atencao_por_token": analise1.atencao_por_token,
            },
            "frase_2": {
                "tokens": analise2.tokens,
                "matriz_atencao": analise2.matriz_atencao,
                "distribuicao": analise2.distribuicao,
                "atencao_por_token": analise2.atencao_por_token,
            },
        }

    def compare_token_importance(
        self,
        pesos1: FloatArray,
        tokens1: list[str],
        num_tokens1: int,
        pesos2: FloatArray,
        tokens2: list[str],
        num_tokens2: int,
    ) -> dict[str, object]:
        """Compara importância de tokens entre duas sequências.

        Args:
            pesos1: Matriz de atenção da sequência 1.
            tokens1: Tokens da sequência 1.
            num_tokens1: Tokens reais da sequência 1.
            pesos2: Matriz de atenção da sequência 2.
            tokens2: Tokens da sequência 2.
            num_tokens2: Tokens reais da sequência 2.

        Returns:
            Dicionário com métricas de importância para ambas as sequências.
        """
        imp1 = self._calcular_importancia(pesos1, tokens1, num_tokens1)
        imp2 = self._calcular_importancia(pesos2, tokens2, num_tokens2)
        return {"frase_1": imp1.__dict__, "frase_2": imp2.__dict__}

    def _calcular_importancia(
        self,
        pesos_atencao: FloatArray,
        tokens: list[str],
        num_tokens_reais: int,
    ) -> TokenImportanceResult:
        """Calcula métricas de importância por token.

        Args:
            pesos_atencao: Matriz de atenção.
            tokens: Lista de tokens.
            num_tokens_reais: Número de tokens reais.

        Returns:
            TokenImportanceResult com as três métricas de importância.
        """
        pesos_reais: FloatArray = pesos_atencao[
            :num_tokens_reais, :num_tokens_reais
        ]
        importancia_recebida: FloatArray = np.sum(pesos_reais, axis=0)
        importancia_dada: FloatArray = np.sum(pesos_reais, axis=1)
        importancia_combinada = (importancia_recebida + importancia_dada) / 2.0

        return TokenImportanceResult(
            tokens=tokens[:num_tokens_reais],
            importancia_recebida=importancia_recebida.tolist(),
            importancia_dada=importancia_dada.tolist(),
            importancia_combinada=importancia_combinada.tolist(),
        )
