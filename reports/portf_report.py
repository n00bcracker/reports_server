import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from config import DOWNLOAD_FOLDER, ORACLE_USERNAME, ORACLE_PASSWORD, ORACLE_TNS, POTRFOLIO_TABLE
import sqlalchemy as sa

datalab_auth = (ORACLE_USERNAME, ORACLE_PASSWORD, ORACLE_TNS)

# Список столбцов для агрегатных функций
group_col = ['req_client',]
cols_count_aggr = ['client_key']
cols_ratio_aggr = ['active', 'is_current', 'block_ufm', 'tax_restrictions', 'sp_fast_growth', 'sp_all_world',
                    'sp_open_opportunities', 'sp_first_step', 'sp_own_business', 'sp_business_package', 'corp_card',
                    'factoring', 'nkl', 'nso', 'electron_guar', 'business_prepaid', 'overdraft', 'deposit',
                    'deposit_gt_3m', 'deposit_gt_5m', 'vkl', 'dvs', 'letters_of_credit', 'credit', 'rko', 'rko_cur',
                    'warranty', 'bss', 'inkass', 'provision_get_pledge', 'sms', 'insurance_contract', 'acct_cur',
                    'conversion', 'npl', 'flag_of_strategic_client', 'zpp', 'ved', 'val_control', 'mobile_bank', 
                    'acquiring', 'flag_phone_number', 'flag_phone_mobile', 'flag_email', 'flag_dbo', 'flag_web_site_org',
                    'consent_bki', 'consent_advertising_messages', 'state_procurements', 'block_ufm_next_3_month',
                    'down_amt_next_month', 'tendency_credit_products', 'tendency_zpp_products',
                    'tendency_acquiring_products', 'tendency_corporate_card', 'tendency_client_escape',
                    'tendency_comp_liq', 'down_active_model', 'giving_out_loan', 'receiving_loan', 'in_transfer_own_money',
                    'out_transfer_own_money',  'pay_salary', 'increase_capital', 'decrease_capital', 'sale_share', 
                    'black_sale', 'covid_19',]
cols_ratio_notnull_aggr = ['business_liquidation_date', 'sup_manager', 'group_id', 'other_bank_name',]
ie_ratio_aggr = ['ip_ul',]
cols_avg_aggr = ['life_span_in_bank', 'life_span_business', 'revenue', 'coowner_cnt', 'cnt_aff_comp', 'amount_dvs', 
                    'amount_dvs_mid', 'amount_deposit', 'amount_nso', 'cnt_out_transactions_per_month',
                    'amt_transactions_per_month', 'cnt_transactions_in_month', 'cnt_in_transactions_per_month', 
                    'cash_withdrawal_amount', 'cash_input_amount', 'cash_withdrawal_atm', 'cash_input_atm',
                    'cash_withdrawal_adm', 'commission_income', 'avg_commis_income_3m', 'revenue_mod', 'revenue_calc',
                    'sum_revenue_group', 'count_company_in_group', 'count_company_clients_bank', 'cnt_employees',
                    'num_out_trans_all_life', 'num_contr', 'life_time_mod', 'ltv_rur', 'wallet_share',]


def read_sql_query(auth_data, sql_query, params=None):
    login, password, tns = auth_data[0], auth_data[1], auth_data[2]
    conn_str = 'oracle+cx_oracle://' + login + ':' + password + '@' + tns
    oracle_db = sa.create_engine(conn_str, encoding='utf-8', max_identifier_length=128)

    conn = oracle_db.connect()
    dataframe = pd.read_sql_query(sql_query, conn, params=params)
    conn.close()
    
    return dataframe

def load_request_clients(filename):

    request_clients = pd.read_excel(filename, dtype={'ИНН' : np.object, 'КПП' : np.object})
    request_clients.columns = [x.lower() for x in request_clients.columns]
    request_clients = request_clients.rename(columns={'инн' : 'inn', 'кпп' : 'kpp'})
    request_clients.loc[request_clients.inn.notna(), 'inn'] = request_clients.loc[request_clients.inn.notna(), 'inn']\
                                                                    .astype(str)
    request_clients.loc[request_clients.kpp.notna(), 'kpp'] = request_clients.loc[request_clients.kpp.notna(), 'kpp']\
                                                                    .astype(str)
    return request_clients

def chi_sqr_test(rates, counts):
    positive_values = np.round(rates * counts)
    obsrv_values = np.vstack([positive_values, counts - positive_values])
    if np.all(obsrv_values > 10):
        pvalue = stats.chi2_contingency(obsrv_values)[1]
    else:
        pvalue = stats.fisher_exact(obsrv_values)[1]
    return pvalue

def f_test(row):
    std1 = row[0]
    std2 = row[1]
    cnt1 = row[2]
    cnt2 = row[3]
    df1 = cnt1 - 1
    df2 = cnt2 - 1
    cvar1 = std1**2 * cnt1 / df1
    cvar2 = std2**2 * cnt2 / df2
    if cvar1 < cvar2:
        pvalue = 1 - stats.f.cdf(cvar2 / cvar1, df2, df1)
    else:
        pvalue = 1 - stats.f.cdf(cvar1 / cvar2, df1, df2) # когда отклоняем, то дисперсия выборки больше дисперсии остальных
    return pvalue

def t_test(row):
    mean1 = row[0]
    mean2 = row[1]
    std1 = row[2]
    std2 = row[3]
    cnt1 = row[4]
    cnt2 = row[5]
    f_pvalue = row[6]
    if f_pvalue < 0.05:
        pvalue = stats.ttest_ind_from_stats(mean1, std1, cnt1, mean2, std2, cnt2, equal_var=False)[1]
    else:
        pvalue = stats.ttest_ind_from_stats(mean1, std1, cnt1, mean2, std2, cnt2, equal_var=True)[1]
    return pvalue

def comment_variance(row):
    std1 = row[0]
    std2 = row[1]
    cnt1 = row[2]
    cnt2 = row[3]
    f_pvalue = row[4]
    
    cvar1 = std1**2 * cnt1 / (cnt1 - 1)
    cvar2 = std2**2 * cnt2 / (cnt2 -1)
    if f_pvalue < 0.05 and cvar1 > cvar2:
        comment = 'Отличие средних значений может быть нерепрезентативным из-за бо\u0301льшего разброса признака среди выборки'
    else:
        comment = np.nan
    return comment

def ratio_aggr_statictcs(aggr_portf, group_counts):
    aggr_portf = aggr_portf.transpose()
    aggr_portf.columns.name = None
    aggr_portf = aggr_portf.rename(columns={True : 'request_clients_ratio', False : 'other_clients_ratio'})
    aggr_portf = aggr_portf.loc[:, ['request_clients_ratio', 'other_clients_ratio']]

    aggr_portf.loc[:, 'p_value'] = aggr_portf.apply(chi_sqr_test, axis=1, args=(group_counts,))
    aggr_portf = aggr_portf.loc[aggr_portf.p_value < 0.2, :]
    aggr_portf['stat_significance'] = (1 - aggr_portf.p_value) * 100
    aggr_portf = aggr_portf.drop(columns=['p_value',])
    return aggr_portf

def avg_aggr_statictcs(aggr_portf):
    aggr_portf = aggr_portf.transpose()
    aggr_portf = aggr_portf.unstack()
    aggr_portf = aggr_portf.dropna(axis=0, how='any')
    aggr_portf.columns = ['other_clients_mean', 'other_clients_std', 'other_clients_cnt', 'request_clients_mean',
                                    'request_clients_std', 'request_clients_cnt']
    aggr_portf = aggr_portf.loc[:, ['request_clients_mean', 'other_clients_mean', 'request_clients_std',
                                                    'other_clients_std', 'request_clients_cnt', 'other_clients_cnt']]

    aggr_portf['f_test_pvalue'] = aggr_portf.loc[:, ['request_clients_std', 'other_clients_std',
                                                    'request_clients_cnt', 'other_clients_cnt']].apply(f_test, axis=1)
    aggr_portf['t_test_pvalue'] = aggr_portf.loc[:, ['request_clients_mean', 'other_clients_mean',
                                                'request_clients_std', 'other_clients_std', 'request_clients_cnt',
                                                'other_clients_cnt', 'f_test_pvalue']].apply(t_test, axis=1)
    aggr_portf['comment'] = aggr_portf.loc[:, ['request_clients_std', 'other_clients_std',
                                    'request_clients_cnt', 'other_clients_cnt', 'f_test_pvalue']].apply(comment_variance)

    aggr_portf = aggr_portf.loc[aggr_portf.t_test_pvalue < 0.2, ['request_clients_mean', 'other_clients_mean',
                                                                't_test_pvalue', 'comment']]
    aggr_portf['stat_significance'] = (1 - aggr_portf.t_test_pvalue) * 100
    aggr_portf = aggr_portf.drop(columns=['t_test_pvalue',])
    return aggr_portf

def make_portf_cmp_report(filename, only_active=False):
    sql_query = f"""
                    select *
                        from {POTRFOLIO_TABLE} t
                            where 1=1
                            and t.ddate = (select max(ddate) from {POTRFOLIO_TABLE})
                """

    portf = read_sql_query(datalab_auth, sql_query)

    # Подготавливаем столбцы, заполняем пропуски
    portf.loc[:, cols_ratio_aggr] = portf.loc[:, cols_ratio_aggr].astype(float)
    portf.loc[:, cols_ratio_aggr] = portf.loc[:, cols_ratio_aggr].fillna(value=0)
    portf.loc[:, cols_ratio_notnull_aggr] = portf.loc[:, cols_ratio_notnull_aggr]\
                                                        .applymap(lambda x : 0 if pd.isnull(x) else 1)
    portf.loc[:, ie_ratio_aggr] = portf.loc[:, ie_ratio_aggr].applymap(lambda x : 0 if x == 'ЮЛ' else 1)
    portf.loc[:, cols_avg_aggr] = portf.loc[:, cols_avg_aggr].astype(float)

    # Загружаем список клиентов для сравнения
    request_clients = load_request_clients(filename)
    if request_clients.client_key.isna().all():
        if request_clients.inn.notna().any():
            request_clients = request_clients.drop(columns=['client_key',])
            request_clients_with_kpp = request_clients.loc[request_clients.kpp.notna(), ['inn', 'kpp']]
            request_clients_with_kpp = request_clients_with_kpp.merge(portf, on=['inn', 'kpp'])\
                                                                    .loc[:, ['inn', 'kpp', 'client_key']]

            request_clients_with_inn = request_clients.loc[~request_clients.kpp.notna(), ['inn',]]
            request_clients_with_inn = request_clients_with_inn.merge(portf, on=['inn',])\
                                                                .loc[:, ['inn', 'kpp', 'client_key']]
            request_clients = pd.concat([request_clients_with_kpp, request_clients_with_inn])
    else:
        print('Отсутствуют ИНН')

    cnt_dupl_clients = request_clients.duplicated(subset=['client_key',]).sum()
    dupl_rate = cnt_dupl_clients / request_clients.shape[0]

    # Маркируем клиентов в портрете на принадлежность к выборке
    portf['req_client'] = portf.client_key.isin(request_clients.client_key)

    # Выбираем с какой группой будем сравнивать
    if only_active:
        portf = portf.loc[portf.req_client | (portf.active == 1), :]

    all_cols = group_col + cols_count_aggr + cols_ratio_aggr + cols_ratio_notnull_aggr + ie_ratio_aggr + cols_avg_aggr
    portf = portf.loc[:, all_cols]

    # Агрегируем значения
    gr_portf = portf.groupby(by=group_col)

    ratio_aggr_portf = gr_portf[cols_ratio_aggr + cols_ratio_notnull_aggr + ie_ratio_aggr].agg(np.mean)
    avg_std_aggr_portf = gr_portf[cols_avg_aggr].agg([np.mean, np.std, 'count'])
    count_aggr_portf = gr_portf[cols_count_aggr].agg('count')

    count_aggr_portf = count_aggr_portf.transpose()
    count_aggr_portf.columns.name = None
    count_aggr_portf = count_aggr_portf.rename(index={'client_key' : 'count'},
                                               columns={True : 'request_clients', False : 'other_clients'})
    count_aggr_portf = count_aggr_portf.loc[:, ['request_clients', 'other_clients']]
    counts = count_aggr_portf.loc['count', ['request_clients', 'other_clients']].values

    # Считаем статистики для доли клиентов в выборке
    report_ratio = ratio_aggr_statictcs(ratio_aggr_portf, counts)

    # Считаем статистики для числовых показателей выборки
    avg_report = avg_aggr_statictcs(avg_std_aggr_portf)

    # Таблица с долями клиентов среди/вне выборки
    report_ratio = report_ratio.rename(columns={'request_clients_ratio' : 'Знач. показателя для клиентов из выборки',
                                                'other_clients_ratio' : 'Знач. показателя для остальных клиентов',
                                                'stat_significance' : 'Уровень достоверности отличия, %'})

    # Таблица со средним по показателям клиентов среди/вне выборки
    avg_report = avg_report.rename(columns={'request_clients_mean' : 'Знач. показателя для клиентов из выборки',
                                                'other_clients_mean' : 'Знач. показателя для остальных клиентов',
                                                'stat_significance' : 'Уровень достоверности отличия, %',
                                                'comment' : 'Комментарий'})

    report = pd.concat([report_ratio, avg_report])
    report['Соотношение'] = report.loc[:, 'Знач. показателя для клиентов из выборки'] / \
                                report.loc[:, 'Знач. показателя для остальных клиентов']
    report = report.loc[:, ['Знач. показателя для клиентов из выборки', 'Знач. показателя для остальных клиентов',
                               'Соотношение', 'Уровень достоверности отличия, %', 'Комментарий']]
    report = report.sort_values('Соотношение', ascending=False)

    count_aggr_portf = count_aggr_portf.rename(columns={'request_clients' : 'Знач. показателя для клиентов из выборки',
                                                'other_clients' : 'Знач. показателя для остальных клиентов'},
                                                index={'count' : 'Кол-во клиентов'})
    report = pd.concat([count_aggr_portf, report])
    report.loc['Кол-во клиентов', 'Комментарий'] = f'Доля дубликатов: {round(dupl_rate * 100,1)}%'

    # Загрузка словаря перевода названий показателей
    cols_vocab = pd.read_excel('meta/Словарь перевода показателей портрета.xlsx')
    cols_vocab.name_eng = cols_vocab.name_eng.str.lower()
    cols_vocab = cols_vocab.set_index('name_eng')
    cols_vocab = cols_vocab.name_rus

    # Переименовываем показатели на русский
    report = report.rename(index=cols_vocab)

    filename_parts = filename.rsplit('.', 1)
    result_filename = filename_parts[0] + '_result.xlsx'
    full_result_filename = os.path.join(DOWNLOAD_FOLDER, result_filename)
    report.to_excel(full_result_filename, encoding='utf-8')

    return result_filename
