import logging
import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from datetime import datetime, timedelta


def log_strategy(strategy_name, factor):
    """
    创建或添加日志，记录策略的运行过程（如每日交易仓位等）到指定日志文件。
    如果 ./logs/strategies/strategy_name.log 不存在，则创建日志文件；
    如果已存在，则继续在日志文件中追加内容。

    Parameters:
        strategy_name (str): 策略名称，用于指定日志文件名。
        factor (bool): 是否是因子类策略，是因子类则为True。

    Returns:
        logging.Logger: 配置好的日志对象。
    """
    # 确定日志目录路径
    if factor:
        log_dir = "./logs/factors/"
    else:
        log_dir = "./logs/strategies/"
    log_file = os.path.join(log_dir, f"{strategy_name}.log")
    
    # 如果日志目录不存在，则创建目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志对象
    logger = logging.getLogger(strategy_name)
    
    # 避免重复添加 Handler
    if not logger.hasHandlers():
        # 设置日志文件处理器
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 设置日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志对象
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    # 设置日志不重复输出到控制台
    logger.propagate = False

    # 记录日志初始化信息
    if factor:
        logger.info(f"=== Factor '{strategy_name}' Log Start ===")
    else:
        logger.info(f"=== Strategy '{strategy_name}' Log Start ===")

    return logger


def get_trade_day(date, gap):
    """
    返回和 date 相差 gap 的交易日。

    Parameters:
        date (datetime or str): 基准日期，格式为 'YYYY-MM-DD' 或 datetime 对象。
        gap (int): 相差的交易日数量。正数表示未来的交易日，负数表示过去的交易日。

    Returns:
        datetime: 目标交易日的日期。
    """
    # 将 date 转换为 datetime 对象
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    
    # 定义交易日列表（实际使用时应替换为从数据源获取的真实交易日列表）
    trade_days = pd.date_range(start="2000-01-01", end="2030-01-01", freq="B")  # 示例为工作日列表
    trade_days = pd.to_datetime(trade_days)

    # 确保输入的日期在交易日范围内
    if date not in trade_days:
        raise ValueError(f"The given date {date.strftime('%Y-%m-%d')} is not a valid trading day.")

    # 获取当前日期在交易日列表中的位置
    current_index = trade_days.get_loc(date)

    # 计算目标交易日的索引
    target_index = current_index + gap

    # 检查是否超出交易日范围
    if target_index < 0 or target_index >= len(trade_days):
        raise ValueError("The resulting trade day is out of the available trading calendar range.")

    # 返回目标交易日
    return trade_days[target_index].to_pydatetime()


def get_trade_day_list(start, end):
    """
    返回在 start 和 end 之间的交易日列表（闭区间）。

    Parameters:
        start (datetime or str): 起始日期，格式为 'YYYY-MM-DD' 或 datetime 对象。
        end (datetime or str): 结束日期，格式为 'YYYY-MM-DD' 或 datetime 对象。

    Returns:
        list[datetime]: 包含所有交易日的列表。
    """
    # 将 start 和 end 转换为 datetime 对象
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d")
    
    # 验证起始日期是否早于结束日期
    if start > end:
        raise ValueError(f"Start date {start.strftime('%Y-%m-%d')} must be earlier than or equal to end date {end.strftime('%Y-%m-%d')}.")

    # 定义交易日列表（实际使用时应替换为从数据源获取的真实交易日列表）
    trade_days = pd.date_range(start="2000-01-01", end="2030-01-01", freq="B")  # 示例为工作日列表
    trade_days = pd.to_datetime(trade_days)

    # 筛选在 start 和 end 间的交易日
    filtered_trade_days = trade_days[(trade_days >= start) & (trade_days <= end)]

    # 返回交易日列表，转换为 Python 的 datetime 对象
    return list(filtered_trade_days.to_pydatetime())


def align_time_index(data, delta):
    """
    对股票数据进行时间对齐，确保所有股票在指定时间列中有一致的索引。
    缺失的时间点将被填充为 NaN。

    Parameters:
        data (pd.DataFrame): 包含时间列和其他数据列的 DataFrame。
                             必须包含 'symbol' 和时间列（'date' 或 ['date', 'timestamp']）。
        delta (str): 时间频率，例如 '1d', '30m', '10m' 等。

    Returns:
        pd.DataFrame: 填充缺失值后的 DataFrame，时间对齐。
    """
    # 校验输入数据
    if 'symbol' not in data.columns:
        raise ValueError("Input data must contain 'symbol' columns.")
    
    if delta == '1d':
        if 'date' not in data.columns:
            raise ValueError("For daily returns, data must contain a 'date' column.")
        time_columns = ['symbol', 'date']
    else:
        if 'date' not in data.columns or 'timestamp' not in data.columns:
            raise ValueError("For intraday returns, data must contain 'date' and 'timestamp' columns.")
        time_columns = ['symbol', 'date', 'timestamp']

        time_columns = ['symbol', 'date'] if delta == '1d' else ['symbol', 'date', 'timestamp']

    # 确保时间列为 datetime 格式
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    # 获取唯一的时间列和股票代码
    unique_symbols = data['symbol'].unique()
    unique_dates = data['date'].unique()
    unique_timestamps = data['timestamp'].unique() if 'timestamp' in data.columns else None

    # 创建完整时间索引
    if delta == '1d':
        all_time_index = pd.DataFrame(
            list(itertools.product(unique_symbols, unique_dates)),
            columns=['symbol', 'date']
        )
    else:
        all_time_index = pd.DataFrame(
            list(itertools.product(unique_symbols, unique_dates, unique_timestamps)),
            columns=['symbol', 'date', 'timestamp']
        )

    # 将时间索引与原始数据对齐
    data = pd.merge(all_time_index, data, on=time_columns, how='left')

    # 按时间排序
    data = data.sort_values(by=time_columns).reset_index(drop=True)

    return data


def calculate_returns(data, delta, lag):
    """
    计算指定滞后期的收益率序列。

    此函数会首先对每只股票的数据进行时间对齐处理，即确保所有股票在指定的时间频率下有一致的时间列。
    如果某些时间点的数据缺失（没有对应行），会补充这些时间点并将其值填充为 NaN。随后，使用滞后期计算收益率，
    并对计算过程中产生的 NaN 值进行填充，默认填充为 0。

    Parameters:
        data (pd.DataFrame): 包含时间列和 'close'（收盘价）、'symbol'（股票代码）。
                             数据格式应包含以下列：
                             - 'close'：股票收盘价；
                             - 'symbol'：股票代码；
                             - 'date'：时间列（对于日频数据）；
                             - 'timestamp'（可选）：时间戳列（对于分钟频数据）。
        delta (str): 时间频率。例如：
                     - '1d' 表示日频，时间列为 'date'；
                     - '30m', '10m', '5m', '1m' 表示分钟频，时间列为 ['date', 'timestamp']。
        lag (int): 计算滞后多少期的收益率。

    Returns:
        pd.DataFrame: 包含原始数据及计算后的收益率列，新增列名为 'return_lag_<lag>'。
                      新增列说明：
                      - 'return_lag_<lag>'：计算得到的滞后期收益率。
                      - 如果某些时间点缺失，收益率值在计算过程中会填充为 0。

    特别说明：
        1. 对于时间对齐：
           - 会对每只股票单独生成完整的时间索引；
           - 如果某些时间点缺失（例如某只股票在某日未交易），会插入这些时间点，并将缺失值填充为 NaN。
        2. 对于收益率计算：
           - 使用矢量化操作计算收益率，避免使用 apply 提高性能；
           - 计算后对所有 NaN 值填充为 0，避免计算错误或遗漏。

    示例用法：
    1. 输入日频数据：
        data = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-02', '2022-01-01', '2022-01-03'],
            'symbol': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
            'close': [150, 152, 2800, 2820]
        })
        delta = '1d'
        lag = 1
        result = calculate_returns(data, delta, lag)

    2. 输出包含滞后收益率的结果：
        date       symbol  close  return_lag_1
        2022-01-01   AAPL   150     0.000000
        2022-01-02   AAPL   152     0.013333
        2022-01-01   GOOG  2800     0.000000
        2022-01-03   GOOG  2820     0.007143
    """
    # 校验输入数据
    if 'close' not in data.columns:
        raise ValueError("Input data must contain 'close' columns.")
    
    # 对齐时间索引
    data = align_time_index(data, delta)

    # 计算滞后期收益率
    data[f'return_lag_{lag}'] = (
        data['close'] / data.groupby('symbol')['close'].shift(lag) - 1
    )

    # 填充缺失值
    data[f'return_lag_{lag}'] = data[f'return_lag_{lag}'].fillna(0)

    return data


def cumulative_returns(data, delta):
    """
    计算累计收益率序列。

    Parameters:
        data (pd.DataFrame): 包含时间列和 'close'（收盘价）、'symbol'（股票代码）。
        delta (str): 时间频率，例如 '1d' 或 '30m'。

    Returns:
        pd.DataFrame: 包含原始数据及计算后的累计收益率列，新增列名为 'cumulative_return'。
    """
    # 校验输入数据
    if 'close' not in data.columns:
        raise ValueError("Input data must contain 'close' columns.")
    
    # 如果 delta 为非 '1d'，需要提取每日最后一个时间点的收盘价
    if delta != '1d':
        # 确保时间列为 datetime 格式
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # 提取每日最后一个时间点的数据
        data = (
            data.sort_values(by=['symbol', 'date', 'timestamp'])
            .groupby(['symbol', 'date'])
            .last()
            .reset_index()
        )
    
    # 调用 calculate_returns 生成滞后收益率列
    data = calculate_returns(data, "1d", 1)

    # 使用生成的滞后收益率列计算累计收益率
    cumulative_return_col = 'return_lag_1'
    data['cumulative_return'] = (
        (1 + data[cumulative_return_col])
        .groupby(data['symbol'])
        .cumprod() - 1
    )
    data = data.drop(cumulative_return_col, axis=1)

    return data


def get_factor(factor_name, timestamp, start, end):
    """
    获取本地 ./factors 下所有指定时间段的因子数据（命名格式为 YYYY-MM-DD.parquet 文件）。
    每一个 parquet 文件都是一个 index 为 symbol、值为因子值的 pd.Series。

    Parameters:
        factor_name (str): 因子名称。
        timestamp (str)：因子生成时间，格式为'15:00:00'或者'10:00:00'
        start (datetime or str): 开始日期，格式为 'YYYY-MM-DD' 或 datetime 对象。
        end (datetime or str): 结束日期，格式为 'YYYY-MM-DD' 或 datetime 对象。

    Returns:
        pd.DataFrame: 包含列 symbol (str)、date（str）、factor_name（因子值）。
    """
    # 获取所有交易日
    trade_days = get_trade_day_list(start, end)
    
    # 因子文件所在路径
    factor_dir = f"./factors/{factor_name}/{timestamp}"
    
    if not os.path.exists(factor_dir):
        raise FileNotFoundError(f"Factor directory {factor_dir} does not exist.")
    
    # 初始化一个字典用于存储数据
    factor_data = {'symbol': [], 'date': [], factor_name: []}

    # 遍历每个交易日，尝试读取对应的 parquet 文件
    for trade_day in trade_days:
        date_str = trade_day.strftime("%Y-%m-%d")
        factor_file = os.path.join(factor_dir, f"{date_str}.parquet")

        if os.path.exists(factor_file):
            # 读取因子文件
            factor_series = pd.read_parquet(factor_file)
            
            # 如果读取到的是 DataFrame，转换为 Series
            if isinstance(factor_series, pd.DataFrame):
                factor_series = factor_series.squeeze()
            
            # 确保是 pd.Series
            if not isinstance(factor_series, pd.Series):
                raise ValueError(f"Expected a pd.Series in {factor_file}, but got {type(factor_series)}.")
            
            # 将数据逐步存入字典
            factor_data['symbol'].extend(factor_series.index.tolist())
            factor_data['date'].extend([date_str] * len(factor_series))
            factor_data[factor_name].extend(factor_series.tolist())
        else:
            print(f"Warning: Factor file {factor_file} does not exist. Skipping.")

    # 将字典转换为 DataFrame
    if factor_data['symbol']:
        result = pd.DataFrame(factor_data)
    else:
        raise ValueError("No valid factor data found for the given time range.")

    return result



def get_factors(factor_names, timestamp, start, end):
    """
    获取多个因子数据，合并为一个数据框，方便后续多因子测试。

    Parameters:
        factor_names (list[str]): 因子名称列表。
        timestamp (str)：因子生成时间，格式为'15:00:00'或者'10:00:00'
        start (datetime or str): 开始日期，格式为 'YYYY-MM-DD' 或 datetime 对象。
        end (datetime or str): 结束日期，格式为 'YYYY-MM-DD' 或 datetime 对象。

    Returns:
        pd.DataFrame: 包含列 symbol (str)、date（str）及各因子值的 DataFrame。
    """
    # 初始化空的 DataFrame，用于存储合并结果
    merged_df = pd.DataFrame()

    for factor in tqdm(factor_names):
        # 获取单个因子数据
        df_factor = get_factor(factor, timestamp, start, end)
        
        if merged_df.empty:
            # 如果是第一个因子，初始化 merged_df
            merged_df = df_factor
        else:
            # 按 symbol 和 date 合并因子数据
            merged_df = pd.merge(
                merged_df,
                df_factor,
                on=['symbol', 'date'],
                how='outer'
            )
    
    # 按 symbol 和 date 排序
    merged_df = merged_df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    return merged_df


def sma(data, factor_name, delta, n):
    """
    通过已经计算好的每日因子值，计算每天的前 N 期因子平均值。

    Parameters:
        data (pd.DataFrame): 包含多个因子数据的 DataFrame。
                             必须包含 'symbol'、'date' 和指定的因子列。
        factor_name (str): 因子名称。
        delta (str): 时间频率，例如 '1d' 或 '30m'。
        n (int): 滑动窗口的大小，计算前 N 期因子均值。

    Returns:
        pd.DataFrame: 原始数据基础上新增一列，列名为 'sma_<factor_name>_<n>'，为计算的简单移动均值。
    """
    # 确保数据中包含必要的列
    if 'symbol' not in data.columns or 'date' not in data.columns or factor_name not in data.columns:
        raise ValueError(f"Data must contain 'symbol', 'date', and '{factor_name}' columns.")

    # 对齐时间索引
    data = align_time_index(data, delta)

    # 按 symbol 和 date 排序
    data = data.sort_values(by=['symbol', 'date']).reset_index(drop=True)

    # 计算简单移动平均值
    sma_column_name = f'sma_{factor_name}_{n}'
    data[sma_column_name] = (
        data.groupby('symbol')[factor_name]
        .rolling(window=n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)  # 删除分组后的多级索引
    )

    return data


def transform_signal(data, factor_name, operation, subtype=None, cover=False):
    """
    对因子数据进行各种操作，包括标准化、去极值、Winsorize 和转换等。

    Parameters:
        data (pd.DataFrame): 包含因子数据的 DataFrame。
        factor_name (str): 因子列的名称。
        operation (str): 主要操作类型，可选值包括：
                         - 'normalize': 按每日标准化。
                         - 'deextreme': 去极值处理。
                         - 'winsorize': Winsorize 方法。
                         - 'transform': 因子值变换。
                         - 'fillna': 填充缺失值。
        subtype (str): 子类型，用于细化某些操作。例如：
                       - 对 `deextreme`：可选 'sigma' 或 'quantile'。
                       - 对 `transform`：可选 'log', 'sqrt'。
                       - 对 `fillna`：可选 'median', 'quantile', 'zero'。
        cover (bool): 处理后的因子是否直接覆盖原列。若为 False，则新增列名为 'transformed_<factor_name>_<operation>'。

    Returns:
        pd.DataFrame: 更新后的 DataFrame。

    Example:
        normalized_data = transform_signal(data, factor_name='factor_1', operation='normalize', cover=True)
        sigma_deextremed_data = transform_signal(data, factor_name='factor_1', operation='deextreme', subtype='sigma')
        filled_data = transform_signal(data, factor_name='factor_1', operation='fillna', subtype='median', cover=True)
    """
    # 确保因子列存在
    if factor_name not in data.columns:
        raise ValueError(f"Factor column '{factor_name}' not found in data.")
    
    # 创建处理后的列名
    new_column_name = factor_name if cover else f"transformed_{factor_name}_{operation}"
    
    # 对因子进行操作
    if operation == 'normalize':
        data[new_column_name] = data.groupby('date')[factor_name].transform(lambda x: (x - x.mean()) / x.std())
    
    elif operation == 'deextreme':
        if subtype == 'sigma':
            mean = data.groupby('date')[factor_name].transform('mean')
            std = data.groupby('date')[factor_name].transform('std')
            upper = mean + 3 * std
            lower = mean - 3 * std
            data[new_column_name] = data[factor_name].clip(lower=lower, upper=upper)
        elif subtype == 'quantile':
            lower = data.groupby('date')[factor_name].transform(lambda x: x.quantile(0.01))
            upper = data.groupby('date')[factor_name].transform(lambda x: x.quantile(0.99))
            data[new_column_name] = data[factor_name].clip(lower=lower, upper=upper)
        else:
            raise ValueError("Invalid subtype for 'deextreme'. Choose 'sigma' or 'quantile'.")
    
    elif operation == 'winsorize':
        lower = data.groupby('date')[factor_name].transform(lambda x: x.quantile(0.05))
        upper = data.groupby('date')[factor_name].transform(lambda x: x.quantile(0.95))
        data[new_column_name] = data[factor_name].clip(lower=lower, upper=upper)
    
    elif operation == 'transform':
        if subtype == 'log':
            data[new_column_name] = np.log1p(data[factor_name])
        elif subtype == 'sqrt':
            data[new_column_name] = np.sqrt(data[factor_name])
        else:
            raise ValueError("Invalid subtype for 'transform'. Choose 'log' or 'sqrt'.")
    
    elif operation == 'fillna':
        if subtype == 'median':
            data[new_column_name] = data.groupby('date')[factor_name].transform(lambda x: x.fillna(x.median()))
        elif subtype == 'quantile':
            data[new_column_name] = data.groupby('date')[factor_name].transform(lambda x: x.fillna(x.quantile(0.5)))
        elif subtype == 'zero':
            data[new_column_name] = data[factor_name].fillna(0)
        else:
            raise ValueError("Invalid subtype for 'fillna'. Choose 'median', 'quantile', or 'zero'.")
    
    else:
        raise ValueError(f"Unsupported operation '{operation}'.")
    
    return data


def market_neutralize(data, factors, market_value_col='market_value_security'):
    """
    对多个因子进行市值中性化处理。

    Parameters:
        data (pd.DataFrame): 包含因子数据和市值列的 DataFrame。
        factors (list[str]): 需要进行市值中性化处理的因子名称列表。
        market_value_col (str): 市值列名称，默认是 'market_value_security'。

    Returns:
        pd.DataFrame: 包含市值中性化后的因子列，新增列名为 'neutralized_<factor_name>'。
    """

    # 检查市值列是否存在
    if market_value_col not in data.columns:
        raise ValueError(f"Market value column '{market_value_col}' not found in data.")

    # 创建市值权重列
    data['market_value_security_weight'] = (
        data[market_value_col] / data[market_value_col].sum() * 100.0
    )

    # 创建正交化权重列
    data['orthogonal_weight'] = data[market_value_col] ** (1 / 3)

    # 初始化一个字典用于存储结果
    neutralized_data = {factor: [] for factor in factors}
    neutralized_data['symbol'] = []
    neutralized_data['date'] = []

    # 回归处理
    for date, group in data.groupby('date'):
        for factor_name in factors:
            # 提取因子和市值权重
            depvar = group[factor_name]
            xvars = group[['market_value_security_weight']].copy()
            xvars['constant'] = 1  # 添加常数项

            # 使用权重回归
            weights = group['orthogonal_weight']
            model = sm.WLS(depvar, xvars, weights=weights)
            results = model.fit()

            # 获取回归残差作为市值中性化后的因子值
            residuals = results.resid

            # 将结果添加到字典中
            neutralized_data[factor_name].extend(residuals.tolist())
        neutralized_data['symbol'].extend(group['symbol'].tolist())
        neutralized_data['date'].extend(group['date'].tolist())

    # 将字典转换为 DataFrame
    neutralized_df = pd.DataFrame(neutralized_data)

    # 将中性化因子合并回原始数据
    for factor_name in factors:
        data[f'neutralized_{factor_name}'] = neutralized_df[factor_name]

    # 删除临时列
    data.drop(columns=['market_value_security_weight', 'orthogonal_weight'], inplace=True)

    return data


def get_data(data_type, start, end):
        """
        获取指定类型的数据。
        :param data_type: 数据类型，例如 'bars', 'balance', 'cashflow'。
        :param start: 开始日期，格式为 'YYYY-MM-DD'。
        :param end: 结束日期，格式为 'YYYY-MM-DD'。
        :return: pd.DataFrame，筛选后的数据。
        """
        file_map = {
            "bars": "stk_daily-2020_2023.feather",
            "balance": "stk_fin_balance-2020_2023.feather",
            "cashflow": "stk_fin_cashflow-2020_2023.feather",
            "income": "stk_fin_income-2020_2023.feather",
        }

        if data_type not in file_map:
            raise ValueError(f"Unsupported data type: {data_type}. Available types: {list(file_map.keys())}.")

        # file_path = os.path.join("./data", file_map[data_type])
        # 获取freetest所在的绝对路径
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        file_path = os.path.join(base_path, file_map[data_type])

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 读取 Feather 文件
        data = pd.read_feather(file_path).rename(columns={'stk_id': 'symbol'})

        # 确保日期列为 datetime 格式
        data["date"] = pd.to_datetime(data["date"])

        # 筛选指定日期范围
        filtered_data = data[(data["date"] >= start) & (data["date"] <= end)].reset_index(drop=True)
        return filtered_data


def read_data(file_name, **kwargs):
        """
        读取本地 ./data 文件夹中的数据，根据文件类型自动选择适当的读取方法。

        Parameters:
            file_name (str): 文件名，包括扩展名（例如 'example.csv', 'example.parquet'）。
            **kwargs: 传递给 pandas 对应读取函数的额外参数。

        Returns:
            pd.DataFrame: 读取的数据。
        
        Raises:
            FileNotFoundError: 如果指定的文件不存在。
            ValueError: 如果文件类型不被支持。
        """
        # 定义数据文件夹路径
        data_dir = "./data"
        file_path = os.path.join(data_dir, file_name)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_name}' does not exist in the './data' folder.")
        
        # 获取文件扩展名
        _, file_extension = os.path.splitext(file_name)
        
        # 根据文件扩展名选择读取方法
        if file_extension == ".csv":
            data = pd.read_csv(file_path, **kwargs)
        elif file_extension == ".parquet":
            data = pd.read_parquet(file_path, **kwargs)
        elif file_extension == ".xlsx" or file_extension == ".xls":
            data = pd.read_excel(file_path, **kwargs)
        elif file_extension == ".json":
            data = pd.read_json(file_path, **kwargs)
        elif file_extension == ".feather":
            data = pd.read_feather(file_path, **kwargs)
        elif file_extension == ".stata":
            data = pd.read_stata(file_path, **kwargs)
        elif file_extension == ".pickle" or file_extension == ".pkl":
            data = pd.read_pickle(file_path, **kwargs)
        elif file_extension == ".h5" or file_extension == ".hdf5":
            data = pd.read_hdf(file_path, **kwargs)
        elif file_extension == ".orc":
            data = pd.read_orc(file_path, **kwargs)
        elif file_extension == ".txt":
            data = pd.read_csv(file_path, delimiter="\t", **kwargs)
        elif file_extension == ".xml":
            data = pd.read_xml(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: '{file_extension}'. Supported types are: .csv, .parquet, .xlsx, .json, .html, .feather, .stata, .pickle, .h5, .orc, .txt, .xml.")
        
        return data
    