"""Testes para o mecanismo de atenção: self-attention, multi-head e análise de tokens.

Cobre tanto a camada de domínio (core/) quanto os endpoints HTTP (routers/attention.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.multi_head_attention import MultiHeadAttention
from core.transformer_simulator import TransformerSimulator


# ---------------------------------------------------------------------------
# Fixtures auxiliares
# ---------------------------------------------------------------------------

D_MODEL_PADRAO: int = 64
SEQ_PADRAO: list[str] = ["o", "gato", "vê", "o", "rato"]
SEQ_ALTERNATIVA: list[str] = ["transformers", "aprendem", "representações"]


def _criar_simulador_com_embeddings(
    tokens: list[str],
    d_model: int = D_MODEL_PADRAO,
) -> tuple[TransformerSimulator, np.ndarray, list[str], int]:
    """Cria simulador e calcula embeddings finais para uso nos testes unitários."""
    seq_length = max(10, len(tokens))
    simulador = TransformerSimulator(d_model=d_model, seq_length=seq_length)
    resultado_emb = simulador.compute_embeddings(tokens)
    embeddings = np.array(resultado_emb.embeddings_finais)
    return simulador, embeddings, resultado_emb.tokens, resultado_emb.num_tokens_reais


# ---------------------------------------------------------------------------
# Testes unitários — Self-Attention (TransformerSimulator)
# ---------------------------------------------------------------------------


class TestSelfAttentionPropriedadesSoftmax:
    """Verifica propriedades matemáticas do softmax na matriz de pesos de atenção."""

    def test_linhas_dos_pesos_somam_1(self) -> None:
        """Cada linha de pesos_atencao deve somar exatamente 1 (propriedade do softmax)."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        pesos = np.array(resultado.pesos_atencao)
        somas_por_linha = pesos.sum(axis=-1)

        np.testing.assert_allclose(
            somas_por_linha,
            np.ones(pesos.shape[0]),
            atol=1e-6,
            err_msg="Cada linha de pesos_atencao deve somar 1 (softmax).",
        )

    def test_linhas_dos_tokens_reais_somam_1(self) -> None:
        """A submatriz exibida (tokens reais x tokens reais) soma 1 por query.

        Garante que o padding não recebe atenção: o frontend mostra só os tokens reais.
        """
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        pesos = np.array(resultado.pesos_atencao)
        np.testing.assert_allclose(
            pesos[:num_reais, :num_reais].sum(axis=-1), np.ones(num_reais), atol=1e-6
        )
        assert np.all(pesos[:, num_reais:] == 0.0), "Keys de padding devem ter peso 0."

    def test_pesos_estao_no_intervalo_zero_um(self) -> None:
        """Todos os pesos de atenção devem estar no intervalo [0, 1]."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        pesos = np.array(resultado.pesos_atencao)

        assert pesos.min() >= 0.0, "Pesos de atenção não podem ser negativos."
        assert pesos.max() <= 1.0 + 1e-9, "Pesos de atenção não podem exceder 1."


class TestShapesQKV:
    """Verifica as dimensões das matrizes Q, K e V."""

    def test_q_shape_correto(self) -> None:
        """Q deve ter shape (seq_length, d_model)."""
        seq_length = max(10, len(SEQ_PADRAO))
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        q = np.array(resultado.Q)
        assert q.shape == (seq_length, D_MODEL_PADRAO), (
            f"Shape de Q esperado ({seq_length}, {D_MODEL_PADRAO}), obtido {q.shape}."
        )

    def test_k_shape_correto(self) -> None:
        """K deve ter shape (seq_length, d_model)."""
        seq_length = max(10, len(SEQ_PADRAO))
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        k = np.array(resultado.K)
        assert k.shape == (seq_length, D_MODEL_PADRAO), (
            f"Shape de K esperado ({seq_length}, {D_MODEL_PADRAO}), obtido {k.shape}."
        )

    def test_v_shape_correto(self) -> None:
        """V deve ter shape (seq_length, d_model)."""
        seq_length = max(10, len(SEQ_PADRAO))
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        v = np.array(resultado.V)
        assert v.shape == (seq_length, D_MODEL_PADRAO), (
            f"Shape de V esperado ({seq_length}, {D_MODEL_PADRAO}), obtido {v.shape}."
        )

    def test_saida_shape_igual_entrada(self) -> None:
        """A saída do self-attention deve ter o mesmo shape dos embeddings de entrada."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado = simulador.compute_attention_step_by_step(emb, tokens, num_reais)

        saida = np.array(resultado.saida)
        assert saida.shape == emb.shape, (
            f"Shape da saída {saida.shape} deve ser igual ao da entrada {emb.shape}."
        )


class TestPadroesDeAtencaoDiferentes:
    """Verifica que tokens distintos produzem padrões de atenção distintos."""

    def test_sequencias_diferentes_geram_padroes_diferentes(self) -> None:
        """Duas sequências distintas devem produzir matrizes de atenção diferentes."""
        sim1, emb1, tok1, n1 = _criar_simulador_com_embeddings(SEQ_PADRAO)
        sim2, emb2, tok2, n2 = _criar_simulador_com_embeddings(SEQ_ALTERNATIVA)

        res1 = sim1.compute_attention_step_by_step(emb1, tok1, n1)
        res2 = sim2.compute_attention_step_by_step(emb2, tok2, n2)

        pesos1 = np.array(res1.pesos_atencao)[:n1, :n1]
        pesos2 = np.array(res2.pesos_atencao)[:n2, :n2]

        # As formas podem diferir; basta que os padrões não sejam idênticos
        # Compara via a norma de Frobenius do sub-bloco comum
        min_n = min(n1, n2)
        bloco1 = pesos1[:min_n, :min_n]
        bloco2 = pesos2[:min_n, :min_n]

        diferenca = np.linalg.norm(bloco1 - bloco2)
        assert diferenca > 1e-6, (
            "Sequências diferentes devem produzir padrões de atenção distintos."
        )


# ---------------------------------------------------------------------------
# Testes unitários — MultiHeadAttention
# ---------------------------------------------------------------------------


class TestMultiHeadAttentionValidacao:
    """Testa validação de parâmetros no construtor de MultiHeadAttention."""

    def test_d_model_nao_divisivel_por_num_heads_levanta_value_error(self) -> None:
        """Deve lançar ValueError quando d_model não é divisível por num_heads."""
        with pytest.raises(ValueError, match="divisível"):
            MultiHeadAttention(d_model=64, num_heads=3)

    def test_parametros_validos_nao_levantam_excecao(self) -> None:
        """Combinações válidas de d_model e num_heads não devem lançar exceção."""
        # d_model=64 é divisível por 1, 2, 4, 8, 16
        for num_heads in (1, 2, 4, 8):
            MultiHeadAttention(d_model=64, num_heads=num_heads)


class TestMultiHeadAttentionCabecasDistintas:
    """Verifica que cada cabeça de atenção aprende padrões diferentes."""

    def test_cabecas_tem_pesos_distintos(self) -> None:
        """Cabeças individuais não devem ter matrizes de pesos idênticas."""
        mha = MultiHeadAttention(d_model=64, num_heads=4, seq_length=10)
        _, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)

        resultado = mha.compute_multi_head_attention(emb, tokens, num_reais)

        # Compara cada par de cabeças adjacentes
        for i in range(len(resultado.cabecas) - 1):
            pesos_i = np.array(resultado.cabecas[i].pesos_atencao)
            pesos_j = np.array(resultado.cabecas[i + 1].pesos_atencao)
            diferenca = np.linalg.norm(pesos_i - pesos_j)
            assert diferenca > 1e-9, (
                f"Cabeça {i} e cabeça {i+1} não devem ter pesos idênticos."
            )


class TestMultiHeadAttentionMascaraPadding:
    """O padding não pode receber atenção em nenhuma cabeça."""

    def test_linhas_dos_tokens_reais_somam_1_em_cada_cabeca(self) -> None:
        mha = MultiHeadAttention(d_model=64, num_heads=4, seq_length=10)
        _, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)

        resultado = mha.compute_multi_head_attention(emb, tokens, num_reais)

        for cabeca in resultado.cabecas:
            pesos = np.array(cabeca.pesos_atencao)
            np.testing.assert_allclose(
                pesos[:num_reais, :num_reais].sum(axis=-1), np.ones(num_reais), atol=1e-6
            )


class TestMultiHeadAttentionShapeSaida:
    """Verifica a dimensão da saída final do multi-head attention."""

    def test_saida_final_shape_seq_length_por_d_model(self) -> None:
        """saida_final deve ter shape (seq_length, d_model)."""
        d_model = 64
        num_heads = 4
        seq_length = max(10, len(SEQ_PADRAO))

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seq_length=seq_length)
        _, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO, d_model)

        resultado = mha.compute_multi_head_attention(emb, tokens, num_reais)

        saida_final = np.array(resultado.saida_final)
        assert saida_final.shape == (seq_length, d_model), (
            f"saida_final shape esperado ({seq_length}, {d_model}), "
            f"obtido {saida_final.shape}."
        )

    def test_numero_de_cabecas_retornado_e_correto(self) -> None:
        """O número de cabeças em MultiHeadAttentionResult deve refletir num_heads."""
        num_heads = 4
        mha = MultiHeadAttention(d_model=64, num_heads=num_heads, seq_length=10)
        _, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)

        resultado = mha.compute_multi_head_attention(emb, tokens, num_reais)

        assert len(resultado.cabecas) == num_heads, (
            f"Esperado {num_heads} cabeças, obtido {len(resultado.cabecas)}."
        )


# ---------------------------------------------------------------------------
# Testes unitários — AttentionFlow e TokenImportance
# ---------------------------------------------------------------------------


class TestAttentionFlow:
    """Verifica propriedades do resultado de compute_attention_flow."""

    def test_pesos_sao_probabilidades_validas(self) -> None:
        """Os pesos do fluxo de atenção devem estar no intervalo [0, 1].

        Com o padding mascarado, a linha restrita aos tokens reais soma 1.
        """
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado_att = simulador.compute_attention_step_by_step(emb, tokens, num_reais)
        pesos_matriz = np.array(resultado_att.pesos_atencao)

        flow = simulador.compute_attention_flow(
            pesos_matriz, tokens, indice_token=0, num_tokens_reais=num_reais
        )

        pesos = np.array(flow.pesos)
        assert pesos.min() >= 0.0, "Pesos do fluxo não podem ser negativos."
        assert pesos.max() <= 1.0 + 1e-9, "Pesos do fluxo não podem exceder 1."
        assert abs(pesos.sum() - 1.0) <= 1e-6, "Os pesos do fluxo devem somar 1."

    def test_conexoes_significativas_tem_peso_acima_de_cinco_porcento(self) -> None:
        """Todas as conexoes_significativas devem ter peso > 0.05."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado_att = simulador.compute_attention_step_by_step(emb, tokens, num_reais)
        pesos_matriz = np.array(resultado_att.pesos_atencao)

        flow = simulador.compute_attention_flow(
            pesos_matriz, tokens, indice_token=0, num_tokens_reais=num_reais
        )

        for conexao in flow.conexoes_significativas:
            assert float(conexao["peso"]) > 0.05, (
                "Conexões significativas devem ter peso > 0.05."
            )


class TestTokenImportance:
    """Verifica que _calcular_importancia retorna o número correto de valores."""

    def test_numero_de_valores_igual_num_tokens_reais(self) -> None:
        """As listas de importância devem ter exatamente num_tokens_reais elementos."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado_att = simulador.compute_attention_step_by_step(emb, tokens, num_reais)
        pesos = np.array(resultado_att.pesos_atencao)

        importancia = simulador._calcular_importancia(pesos, tokens, num_reais)

        assert len(importancia.importancia_recebida) == num_reais, (
            "importancia_recebida deve ter um valor por token real."
        )
        assert len(importancia.importancia_dada) == num_reais, (
            "importancia_dada deve ter um valor por token real."
        )
        assert len(importancia.importancia_combinada) == num_reais, (
            "importancia_combinada deve ter um valor por token real."
        )

    def test_importancia_combinada_e_media_das_outras_duas(self) -> None:
        """importancia_combinada deve ser a média aritmética das outras métricas."""
        simulador, emb, tokens, num_reais = _criar_simulador_com_embeddings(SEQ_PADRAO)
        resultado_att = simulador.compute_attention_step_by_step(emb, tokens, num_reais)
        pesos = np.array(resultado_att.pesos_atencao)

        importancia = simulador._calcular_importancia(pesos, tokens, num_reais)

        recebida = np.array(importancia.importancia_recebida)
        dada = np.array(importancia.importancia_dada)
        combinada = np.array(importancia.importancia_combinada)

        np.testing.assert_allclose(
            combinada,
            (recebida + dada) / 2.0,
            atol=1e-9,
            err_msg="importancia_combinada deve ser (recebida + dada) / 2.",
        )


# ---------------------------------------------------------------------------
# Testes de integração — Endpoints HTTP
# ---------------------------------------------------------------------------


class TestEndpointSelfAttention:
    """Testes para POST /api/attention/self-attention."""

    def test_retorna_q_k_v_e_pesos(self, client: TestClient) -> None:
        """Resposta deve conter Q, K, V e pesos_atencao."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/self-attention", json=payload)

        assert response.status_code == 200
        data = response.json()

        for campo in ("Q", "K", "V", "pesos_atencao", "scores_escalados", "saida"):
            assert campo in data, f"Campo '{campo}' ausente na resposta."

    def test_pesos_de_atencao_linhas_somam_1(self, client: TestClient) -> None:
        """As linhas de pesos_atencao retornadas pelo endpoint devem somar 1."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/self-attention", json=payload)

        assert response.status_code == 200
        pesos = np.array(response.json()["pesos_atencao"])
        somas = pesos.sum(axis=-1)

        np.testing.assert_allclose(
            somas,
            np.ones(pesos.shape[0]),
            atol=1e-6,
            err_msg="Linhas de pesos_atencao devem somar 1.",
        )

    def test_num_tokens_reais_reflete_tamanho_da_entrada(self, client: TestClient) -> None:
        """num_tokens_reais deve ser igual ao comprimento da lista enviada."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/self-attention", json=payload)

        assert response.status_code == 200
        assert response.json()["num_tokens_reais"] == len(SEQ_PADRAO)

    def test_passos_explicados_esta_presente(self, client: TestClient) -> None:
        """Resposta deve incluir o campo passos_explicados com 5 passos."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/self-attention", json=payload)

        assert response.status_code == 200
        passos = response.json()["passos_explicados"]
        assert isinstance(passos, list)
        assert len(passos) == 5


class TestEndpointMultiHeadAttention:
    """Testes para POST /api/attention/multi-head-attention."""

    def test_retorna_numero_correto_de_cabecas(self, client: TestClient) -> None:
        """O número de cabeças na resposta deve corresponder ao solicitado."""
        num_cabecas = 4
        payload = {"tokens": SEQ_PADRAO, "d_model": 64, "num_cabecas": num_cabecas}
        response = client.post("/api/attention/multi-head-attention", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["num_cabecas"] == num_cabecas
        assert len(data["cabecas"]) == num_cabecas

    def test_saida_final_presente_na_resposta(self, client: TestClient) -> None:
        """Resposta deve conter o campo saida_final."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64, "num_cabecas": 4}
        response = client.post("/api/attention/multi-head-attention", json=payload)

        assert response.status_code == 200
        assert "saida_final" in response.json()

    def test_d_k_calculado_corretamente(self, client: TestClient) -> None:
        """d_k na resposta deve ser d_model // num_cabecas."""
        d_model = 64
        num_cabecas = 4
        payload = {"tokens": SEQ_PADRAO, "d_model": d_model, "num_cabecas": num_cabecas}
        response = client.post("/api/attention/multi-head-attention", json=payload)

        assert response.status_code == 200
        assert response.json()["d_k"] == d_model // num_cabecas


class TestEndpointMultiHeadAttentionInvalido:
    """Testes para validação de d_model e num_cabecas incompatíveis."""

    def test_d_model_nao_divisivel_por_num_cabecas_retorna_422(
        self, client: TestClient
    ) -> None:
        """Quando d_model % num_cabecas != 0, o endpoint deve retornar 422."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64, "num_cabecas": 3}
        response = client.post("/api/attention/multi-head-attention", json=payload)

        assert response.status_code == 422

    def test_mensagem_de_erro_menciona_divisibilidade(
        self, client: TestClient
    ) -> None:
        """O detalhe do erro deve mencionar a restrição de divisibilidade."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64, "num_cabecas": 3}
        response = client.post("/api/attention/multi-head-attention", json=payload)

        assert response.status_code == 422
        detalhe = str(response.json())
        # O router formata o detail como string descritiva
        assert "64" in detalhe or "divisível" in detalhe or "3" in detalhe


class TestEndpointAttentionFlow:
    """Testes para POST /api/attention/attention-flow."""

    def test_retorna_pesos_e_conexoes_significativas(self, client: TestClient) -> None:
        """Resposta deve conter pesos e conexoes_significativas."""
        payload = {"tokens": SEQ_PADRAO, "indice_token": 0, "d_model": 64}
        response = client.post("/api/attention/attention-flow", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "pesos" in data
        assert "conexoes_significativas" in data

    def test_token_alvo_corresponde_ao_indice(self, client: TestClient) -> None:
        """token_alvo deve ser o token no índice solicitado."""
        indice = 2
        payload = {"tokens": SEQ_PADRAO, "indice_token": indice, "d_model": 64}
        response = client.post("/api/attention/attention-flow", json=payload)

        assert response.status_code == 200
        assert response.json()["token_alvo"] == SEQ_PADRAO[indice]

    def test_pesos_estao_no_intervalo_valido(self, client: TestClient) -> None:
        """Os pesos retornados pelo endpoint devem estar em [0, 1] com soma <= 1.

        Os pesos são um slice de uma linha do softmax (apenas tokens reais),
        portanto a soma pode ser menor que 1 quando há padding na sequência.
        """
        payload = {"tokens": SEQ_PADRAO, "indice_token": 0, "d_model": 64}
        response = client.post("/api/attention/attention-flow", json=payload)

        assert response.status_code == 200
        pesos = np.array(response.json()["pesos"])

        assert pesos.min() >= 0.0, "Pesos do fluxo não podem ser negativos."
        assert pesos.max() <= 1.0 + 1e-9, "Pesos do fluxo não podem exceder 1."
        assert pesos.sum() <= 1.0 + 1e-9, (
            "A soma dos pesos não pode exceder 1."
        )

    def test_apenas_um_token_retorna_422(self, client: TestClient) -> None:
        """A requisição com apenas 1 token deve ser rejeitada com 422 (min_length=2)."""
        payload = {"tokens": ["palavra"], "indice_token": 0, "d_model": 64}
        response = client.post("/api/attention/attention-flow", json=payload)

        assert response.status_code == 422


class TestEndpointTokenImportance:
    """Testes para POST /api/attention/token-importance."""

    def test_retorna_tres_listas_de_importancia(self, client: TestClient) -> None:
        """Resposta deve conter importancia_recebida, importancia_dada e importancia_combinada."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/token-importance", json=payload)

        assert response.status_code == 200
        data = response.json()

        for campo in ("importancia_recebida", "importancia_dada", "importancia_combinada"):
            assert campo in data, f"Campo '{campo}' ausente na resposta."

    def test_numero_de_valores_igual_ao_num_tokens_reais(
        self, client: TestClient
    ) -> None:
        """Cada lista de importância deve ter exatamente num_tokens_reais elementos."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/token-importance", json=payload)

        assert response.status_code == 200
        data = response.json()
        n = data["num_tokens_reais"] if "num_tokens_reais" in data else len(SEQ_PADRAO)

        # A resposta não inclui num_tokens_reais diretamente, usamos o len dos tokens
        n_tokens = len(data["tokens"])
        assert len(data["importancia_recebida"]) == n_tokens
        assert len(data["importancia_dada"]) == n_tokens
        assert len(data["importancia_combinada"]) == n_tokens

    def test_token_mais_importante_esta_na_lista_de_tokens(
        self, client: TestClient
    ) -> None:
        """token_mais_importante deve ser um dos tokens retornados."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/token-importance", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["token_mais_importante"] in data["tokens"]

    def test_importancia_combinada_e_media_das_outras_duas(
        self, client: TestClient
    ) -> None:
        """importancia_combinada deve ser (recebida + dada) / 2 para cada token."""
        payload = {"tokens": SEQ_PADRAO, "d_model": 64}
        response = client.post("/api/attention/token-importance", json=payload)

        assert response.status_code == 200
        data = response.json()

        recebida = np.array(data["importancia_recebida"])
        dada = np.array(data["importancia_dada"])
        combinada = np.array(data["importancia_combinada"])

        np.testing.assert_allclose(
            combinada,
            (recebida + dada) / 2.0,
            atol=1e-9,
            err_msg="importancia_combinada deve ser a média de recebida e dada.",
        )
