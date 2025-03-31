from typing import Any, List, Union

import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.metrics import roc_auc_score

from IPython.display import clear_output, display
from statsmodels.stats.outliers_influence import variance_inflation_factor


def read_data(
    file_name: str, file_format: str = 'csv', **kwargs: dict
) -> pd.DataFrame:
    '''Carregar dados em formato colunar.

    Params
    ------
    file_name: str
        Nome do arquivo a ser carregado.
    file_format: str
        Formato do arquivo.
    **kwargs: dict
        Parâmetros do método de leitura.

    Retorna
    -------
    pandas.DataFrame
        Dados carregados.
    '''
    if file_format == 'csv':
        return pd.read_csv(file_name, **kwargs)
    elif file_format == 'parquet':
        return pd.read_parquet(file_name, **kwargs)
    elif file_format in ['excel', 'xls', 'xlsx']:
        with open(file_name, 'rb') as f:
            return pd.read_excel(f, **kwargs)
    else:
        raise NotImplementedError(
            f'Leitor para arquivos no formato {file_format} não implementado.'
        )


def read_yaml(file_name: str) -> dict:
    '''Ler um arquivo YAML.

    Params
    ------
    file_name: str
        Nome do arquivo a ser carregado.

    Retorna
    -------
    dict
        Conteúdo do arquivo.
    '''
    with open(file_name) as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, file_name: str):
    '''Salvar um objeto em formato pickle.

    Params
    ------
    obj: Any
        Objeto a ser salvo.
    file_name: str
        Nome do arquivo a ser gerado.
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def value_counts(data: pd.Series) -> pd.DataFrame:
    '''Criar uma tabela com a contagem de valores de uma variável.

    Params
    ------
    data: pandas.Series
        Variável a ser avaliada.

    Retorna
    -------
    pandas.DataFrame
        Tabela com a contagem e porcentagem de cada valor da variável.
    '''
    return pd.DataFrame({
        'qtd': data.value_counts(
            dropna=False
        ).sort_index().apply(lambda x: f'{x:,d}'),
        '%': data.value_counts(
            dropna=False, normalize=True
        ).sort_index().apply(lambda x: f'{x:.1%}')
    })


def calculate_psi(
    ref_vals: Union[pd.Series, np.ndarray],
    curr_vals: Union[pd.Series, np.ndarray],
    bins: int
) -> float:
    '''Calcular o Índice de Estabilidade Populacional (Population Stability
    Index - PSI) de uma distribuição em relação a outra distribuição de
    referência.

    Params
    ------
    ref_vals: Union[pd.Series, np.ndarray]
        Distribuição de referência.
    curr_vals: Union[pd.Series, np.ndarray]
        Distribuição atual.
    bins: int
        Número de bins para discretizar as distribuições.

    Retorna
    -------
    float
        Índice de Estabilidade Populacional.
    '''
    # Discretizar os valores com base na distribuição de referência
    ref_counts, bin_limits = np.histogram(ref_vals, bins=bins)
    curr_counts, _ = np.histogram(curr_vals, bins=bin_limits)

    ref_percs = ref_counts / ref_counts.sum()
    curr_percs = curr_counts / curr_counts.sum()

    # Corrigir divisão por 0 nas porcentagens
    ref_percs = np.where(ref_percs == 0, 0.00001, ref_percs)
    curr_percs = np.where(curr_percs == 0, 0.00001, curr_percs)

    # Calcular PSI
    psi_values = (curr_percs - ref_percs) * np.log(curr_percs / ref_percs)

    return float(np.sum(psi_values))


def drop_unstable_feats(
    data: pd.DataFrame,
    ym_var: str,
    vars: List[str],
    psi_thr: float,
    psi_bins: int = 10
) -> List[str]:
    '''Criar uma lista de variáveis instáveis de acordo com o PSI.

    Params
    ------
    data: pandas.DataFrame
        Conjunto de dados.
    ym_var: str
        Nome da variável de safra.
    vars: List[str]
        Nomes das variáveis a serem avaliadas.
    psi_thr: float
        Valor máximo de PSI a ser mantido.
    psi_bins: int (default=10)
        Número de bins para discretizar as distribuições.

    Retorna
    -------
    List[str]
        Nomes das variáveis instáveis.
    '''
    drop_vars = []

    ref_ym = data[ym_var].min()

    data_ref = data[data[ym_var] == ref_ym]
    yms = sorted(data[ym_var].unique().tolist())

    for var in vars:
        psi_vals = []

        for ym in yms:
            if ym == ref_ym:
                continue

            data_curr = data[data[ym_var] == ym]
            psi = calculate_psi(data_ref[var], data_curr[var], psi_bins)
            psi_vals.append(psi)

        # Se o PSI da variável for > limite em algum mês, remover
        if max(psi_vals, default=0) > psi_thr:
            drop_vars.append(var)

    return drop_vars


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    '''Calcular o Fator de Inflação da Variância (Variance Inflation
    Factor - VIF) para as variáveis de um conjunto de dados.

    Params
    ------
    data: pandas.DataFrame
        Conjunto de dados.

    Retorna
    -------
    pandas.DataFrame
        Tabela com o VIF de cada variável do conjunto de dados.
    '''
    vif_data = pd.DataFrame()
    vif_data['feature'] = data.columns
    vif_data['vif'] = [
        variance_inflation_factor(data.values, i)
        for i in range(len(data.columns))
    ]

    return vif_data


def drop_colinear_feats(
    data: pd.DataFrame,
    vif_thr: int = 10,
    max_iter_vif: float = np.inf
) -> List[str]:
    '''Criar um lista de variáveis com multicolinearidade de acordo com o VIF.

    As variáveis são removidas de forma iterativa, até que não haja nenhuma
    variável com VIF acima do limite ou o número máximo de variável removidas
    tenha sido removido. A cada iteração, são executados os seguintes passos:

        1. Calcular o VIF de cada variável em relação às demais disponíveis.
        2. Selecionar a variável com maior VIF, se VIF > limite.
        3. Remover a variável selecionada da lista de variáveis disponíveis.

    Params
    ------
    data: pandas.DataFrame
        Conjunto de dados.
    vif_thr: int (default=10)
        Valor máximo de VIF a ser mantido.
    max_iter_vif: float (default=np.inf)
        Número máximo de variáveis a serem removidas.

    Retorna
    -------
    List[str]
        Nome das variáveis com multicolinearidade.
    '''
    drop_vars = []

    max_vif = np.inf
    i = 1

    while max_vif > vif_thr:
        clear_output(wait=True)
        display(f'Processando feature {i}...')

        # Calcular o VIF de todas as variáveis
        vif_data = calculate_vif(data.drop(columns=drop_vars))

        # Obter a variável com maior VIF
        vif_top1 = vif_data.sort_values(by='vif', ascending=False).iloc[0]
        max_vif = vif_top1['vif']

        # Se o VIF da variável for > limite, remover
        if max_vif > vif_thr:
            drop_vars.append(vif_top1['feature'])

        if i > max_iter_vif:
            break

        i += 1

    return drop_vars


def gini(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray]
) -> float:
    '''Calcular o Gini a partir das probabilidade de uma previsão.

    Params
    ------
    y_true: Union[pandas.Series, numpy.ndarray]
        Valores reais da variável.
    y_proba: Union[pandas.Series, numpy.ndarray]
        Probabilidade predita.

    Retorna
    -------
    float
        Gini = 2 x AUC - 1.
    '''
    auc = roc_auc_score(y_true, y_proba)

    return 2 * auc - 1
