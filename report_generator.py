"""
Модуль для генерации HTML-отчётов
Создает интерактивные отчёты с визуализациями и результатами анализа
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from datetime import datetime


def generate_html_report(results):
    """
    Генерация полного HTML-отчёта
    
    Параметры:
    ----------
    results : dict
        Словарь с результатами анализа
    
    Возвращает:
    ----------
    str : HTML-код отчёта
    """
    ts_data = results['ts_data']
    params = results['params']
    
    # Создание HTML-шаблона
    html_template = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчёт по анализу временного ряда</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 40px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            h3 {{
                color: #555;
                margin-top: 25px;
                margin-bottom: 15px;
            }}
            .info-box {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            .info-box p {{
                margin: 8px 0;
                color: #2c3e50;
            }}
            .info-box strong {{
                color: #2980b9;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .stat-card h4 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
                margin: 10px 0;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 14px;
            }}
            .success {{
                color: #27ae60;
                font-weight: bold;
            }}
            .warning {{
                color: #e67e22;
                font-weight: bold;
            }}
            .chart-container {{
                margin: 30px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Отчёт по анализу временного ряда</h1>
            
            <div class="info-box">
                <p><strong>Дата создания:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
                <p><strong>Целевая переменная:</strong> {params['target_column']}</p>
                <p><strong>Период сезонности:</strong> {params['seasonal_period']}</p>
                <p><strong>Модель декомпозиции:</strong> {'Аддитивная' if params['decomposition_model'] == 'additive' else 'Мультипликативная'}</p>
                <p><strong>Размер окна скользящего среднего:</strong> {params['rolling_window']}</p>
                <p><strong>Максимальное количество лагов:</strong> {params['max_lags']}</p>
            </div>
            
            <h2>1. Временной ряд с трендом и скользящим средним</h2>
            <div class="chart-container" id="chart1"></div>
            
            <h2>2. Автокорреляционные функции</h2>
            <div class="chart-container" id="chart2"></div>
            
            <h2>3. Декомпозиция временного ряда</h2>
            <div class="chart-container" id="chart3"></div>
            
            <h2>4. Тесты на стационарность</h2>
            {generate_stationarity_html(results)}
            
            {generate_correlation_html(results) if results.get('correlation_matrix') is not None else ''}
            
            <div class="footer">
                <p>Отчёт создан с использованием Python, Plotly и Streamlit</p>
                <p>© 2025 Time Series Analysis Tool</p>
            </div>
        </div>
        
        <script>
            {generate_chart1_js(results)}
            {generate_chart2_js(results)}
            {generate_chart3_js(results)}
            {generate_correlation_chart_js(results) if results.get('correlation_matrix') is not None else ''}
        </script>
    </body>
    </html>
    """
    
    return html_template


def generate_chart1_js(results):
    """Генерация JavaScript для графика временного ряда"""
    ts_data = results['ts_data']
    rolling_stats = results['rolling_stats']
    params = results['params']
    
    # Преобразование дат в строки
    dates = ts_data['date'].dt.strftime('%Y-%m-%d').tolist()
    values = ts_data['value'].tolist()
    rolling_mean = rolling_stats['rolling_mean'].tolist()
    rolling_std = rolling_stats['rolling_std'].tolist()
    
    js_code = f"""
    var trace1 = {{
        x: {dates},
        y: {values},
        mode: 'lines',
        name: 'Исходный ряд',
        line: {{color: 'lightblue', width: 1}}
    }};
    
    var trace2 = {{
        x: {dates},
        y: {rolling_mean},
        mode: 'lines',
        name: 'Скользящее среднее ({params["rolling_window"]})',
        line: {{color: 'orange', width: 2}}
    }};
    
    var trace3 = {{
        x: {dates},
        y: {rolling_std},
        mode: 'lines',
        name: 'Скользящее стд. откл. ({params["rolling_window"]})',
        line: {{color: 'red', width: 2, dash: 'dash'}}
    }};
    
    var layout1 = {{
        title: 'Временной ряд: {params["target_column"]}',
        xaxis: {{title: 'Дата'}},
        yaxis: {{title: 'Значение'}},
        hovermode: 'x unified',
        height: 500
    }};
    
    Plotly.newPlot('chart1', [trace1, trace2, trace3], layout1);
    """
    
    return js_code


def generate_chart2_js(results):
    """Генерация JavaScript для графиков ACF и PACF"""
    acf = results['acf']
    pacf = results['pacf']
    acf_confint = results['acf_confint']
    pacf_confint = results['pacf_confint']
    
    lags_acf = list(range(len(acf)))
    lags_pacf = list(range(len(pacf)))
    
    acf_upper = acf_confint[:, 1].tolist()
    acf_lower = acf_confint[:, 0].tolist()
    pacf_upper = pacf_confint[:, 1].tolist()
    pacf_lower = pacf_confint[:, 0].tolist()
    
    js_code = f"""
    // ACF
    var trace_acf = {{
        x: {lags_acf},
        y: {acf.tolist()},
        type: 'bar',
        name: 'ACF',
        marker: {{color: 'steelblue'}}
    }};
    
    var trace_acf_upper = {{
        x: {lags_acf},
        y: {acf_upper},
        mode: 'lines',
        name: 'Доверительный интервал',
        line: {{color: 'red', dash: 'dash'}},
        showlegend: false
    }};
    
    var trace_acf_lower = {{
        x: {lags_acf},
        y: {acf_lower},
        mode: 'lines',
        name: 'Доверительный интервал',
        line: {{color: 'red', dash: 'dash'}},
        showlegend: false
    }};
    
    // PACF
    var trace_pacf = {{
        x: {lags_pacf},
        y: {pacf.tolist()},
        type: 'bar',
        name: 'PACF',
        marker: {{color: 'darkorange'}}
    }};
    
    var trace_pacf_upper = {{
        x: {lags_pacf},
        y: {pacf_upper},
        mode: 'lines',
        name: 'Доверительный интервал',
        line: {{color: 'red', dash: 'dash'}},
        showlegend: false
    }};
    
    var trace_pacf_lower = {{
        x: {lags_pacf},
        y: {pacf_lower},
        mode: 'lines',
        name: 'Доверительный интервал',
        line: {{color: 'red', dash: 'dash'}},
        showlegend: false
    }};
    
    var layout2 = {{
        title: 'Автокорреляционные функции (ACF и PACF)',
        grid: {{rows: 1, columns: 2, pattern: 'independent'}},
        xaxis: {{title: 'Лаг', domain: [0, 0.45]}},
        xaxis2: {{title: 'Лаг', domain: [0.55, 1]}},
        yaxis: {{title: 'Корреляция'}},
        yaxis2: {{title: 'Частичная корреляция', anchor: 'x2'}},
        height: 500,
        showlegend: true
    }};
    
    var acf_data = [trace_acf, trace_acf_upper, trace_acf_lower];
    var pacf_data = [trace_pacf, trace_pacf_upper, trace_pacf_lower];
    
    // Добавляем xaxis2 и yaxis2 для PACF
    pacf_data.forEach(function(trace) {{
        trace.xaxis = 'x2';
        trace.yaxis = 'y2';
    }});
    
    var all_data = acf_data.concat(pacf_data);
    
    Plotly.newPlot('chart2', all_data, layout2);
    """
    
    return js_code


def generate_chart3_js(results):
    """Генерация JavaScript для декомпозиции"""
    ts_data = results['ts_data']
    decomp = results['decomposition']
    
    dates = ts_data['date'].dt.strftime('%Y-%m-%d').tolist()
    
    # Обработка NaN значений
    observed = [None if np.isnan(x) else x for x in decomp.observed]
    trend = [None if np.isnan(x) else x for x in decomp.trend]
    seasonal = [None if np.isnan(x) else x for x in decomp.seasonal]
    resid = [None if np.isnan(x) else x for x in decomp.resid]
    
    js_code = f"""
    var trace_obs = {{
        x: {dates},
        y: {observed},
        mode: 'lines',
        name: 'Исходный',
        line: {{color: 'blue'}},
        xaxis: 'x',
        yaxis: 'y'
    }};
    
    var trace_trend = {{
        x: {dates},
        y: {trend},
        mode: 'lines',
        name: 'Тренд',
        line: {{color: 'orange'}},
        xaxis: 'x2',
        yaxis: 'y2'
    }};
    
    var trace_seasonal = {{
        x: {dates},
        y: {seasonal},
        mode: 'lines',
        name: 'Сезонность',
        line: {{color: 'green'}},
        xaxis: 'x3',
        yaxis: 'y3'
    }};
    
    var trace_resid = {{
        x: {dates},
        y: {resid},
        mode: 'lines',
        name: 'Остатки',
        line: {{color: 'red'}},
        xaxis: 'x4',
        yaxis: 'y4'
    }};
    
    var layout3 = {{
        title: 'Декомпозиция временного ряда',
        grid: {{rows: 4, columns: 1, subplots:[['xy'],['x2y2'],['x3y3'],['x4y4']], roworder:'top to bottom'}},
        height: 1000,
        showlegend: false,
        yaxis: {{title: 'Исходный'}},
        yaxis2: {{title: 'Тренд'}},
        yaxis3: {{title: 'Сезонность'}},
        yaxis4: {{title: 'Остатки'}},
        xaxis4: {{title: 'Дата'}}
    }};
    
    Plotly.newPlot('chart3', [trace_obs, trace_trend, trace_seasonal, trace_resid], layout3);
    """
    
    return js_code


def generate_stationarity_html(results):
    """Генерация HTML для результатов тестов на стационарность"""
    adf = results['adf_result']
    kpss = results['kpss_result']
    
    adf_status = "success" if adf['p_value'] < 0.05 else "warning"
    kpss_status = "success" if kpss['p_value'] > 0.05 else "warning"
    
    adf_interpretation = "✅ Ряд стационарный (p < 0.05)" if adf['p_value'] < 0.05 else "⚠️ Ряд нестационарный (p >= 0.05)"
    kpss_interpretation = "✅ Ряд стационарный (p > 0.05)" if kpss['p_value'] > 0.05 else "⚠️ Ряд нестационарный (p <= 0.05)"
    
    html = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <h4>Расширенный тест Дики-Фуллера (ADF)</h4>
            <table>
                <tr>
                    <td><strong>ADF-статистика:</strong></td>
                    <td>{adf['adf_stat']:.4f}</td>
                </tr>
                <tr>
                    <td><strong>p-значение:</strong></td>
                    <td>{adf['p_value']:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Использовано лагов:</strong></td>
                    <td>{adf['lags_used']}</td>
                </tr>
                <tr>
                    <td><strong>Количество наблюдений:</strong></td>
                    <td>{adf['n_obs']}</td>
                </tr>
            </table>
            <h4>Критические значения:</h4>
            <table>
                <tr>
                    <th>Уровень значимости</th>
                    <th>Критическое значение</th>
                </tr>
                <tr>
                    <td>1%</td>
                    <td>{adf['critical_values']['1%']:.4f}</td>
                </tr>
                <tr>
                    <td>5%</td>
                    <td>{adf['critical_values']['5%']:.4f}</td>
                </tr>
                <tr>
                    <td>10%</td>
                    <td>{adf['critical_values']['10%']:.4f}</td>
                </tr>
            </table>
            <p class="{adf_status}"><strong>{adf_interpretation}</strong></p>
        </div>
        
        <div class="stat-card">
            <h4>Тест Квятковского-Филлипса-Шмидта-Шина (KPSS)</h4>
            <table>
                <tr>
                    <td><strong>KPSS-статистика:</strong></td>
                    <td>{kpss['kpss_stat']:.4f}</td>
                </tr>
                <tr>
                    <td><strong>p-значение:</strong></td>
                    <td>{kpss['p_value']:.4f}</td>
                </tr>
                <tr>
                    <td><strong>Использовано лагов:</strong></td>
                    <td>{kpss['lags_used']}</td>
                </tr>
            </table>
            <h4>Критические значения:</h4>
            <table>
                <tr>
                    <th>Уровень значимости</th>
                    <th>Критическое значение</th>
                </tr>
                <tr>
                    <td>10%</td>
                    <td>{kpss['critical_values']['10%']:.4f}</td>
                </tr>
                <tr>
                    <td>5%</td>
                    <td>{kpss['critical_values']['5%']:.4f}</td>
                </tr>
                <tr>
                    <td>2.5%</td>
                    <td>{kpss['critical_values']['2.5%']:.4f}</td>
                </tr>
                <tr>
                    <td>1%</td>
                    <td>{kpss['critical_values']['1%']:.4f}</td>
                </tr>
            </table>
            <p class="{kpss_status}"><strong>{kpss_interpretation}</strong></p>
        </div>
    </div>
    """
    
    return html


def generate_correlation_html(results):
    """Генерация HTML для корреляционной матрицы"""
    html = """
    <h2>5. Корреляционная матрица</h2>
    <div class="chart-container" id="chart4"></div>
    """
    return html


def generate_correlation_chart_js(results):
    """Генерация JavaScript для тепловой карты корреляций"""
    if results.get('correlation_matrix') is None:
        return ""
    
    corr_matrix = results['correlation_matrix']
    
    # Преобразование в списки
    z_values = corr_matrix.values.tolist()
    x_labels = corr_matrix.columns.tolist()
    y_labels = corr_matrix.index.tolist()
    
    # Создание текстовых аннотаций
    text_values = [[f"{val:.2f}" for val in row] for row in z_values]
    
    js_code = f"""
    var data_corr = [{{
        z: {z_values},
        x: {x_labels},
        y: {y_labels},
        type: 'heatmap',
        colorscale: 'RdBu',
        reversescale: true,
        zmid: 0,
        text: {text_values},
        texttemplate: '%{{text}}',
        textfont: {{size: 10}},
        colorbar: {{title: 'Корреляция'}}
    }}];
    
    var layout_corr = {{
        title: 'Тепловая карта корреляций',
        height: 500,
        xaxis: {{side: 'bottom'}},
        yaxis: {{autorange: 'reversed'}}
    }};
    
    Plotly.newPlot('chart4', data_corr, layout_corr);
    """
    
    return js_code

