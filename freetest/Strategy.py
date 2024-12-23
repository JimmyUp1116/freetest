import os
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Callable
from pypfopt import expected_returns, risk_models, EfficientFrontier, HRPOpt
from abc import ABC, abstractmethod

from .Position import Position
from .utils import get_trade_day_list, log_strategy, get_factor

class SignalStrategy:
    def __init__(self, time, start, end, name):
        """
        初始化 SignalStrategy 类。

        Parameters:
            time (str): 因子生成的时间点，例如 "15:00:00"。
            start (str): 因子生成的初始日期，格式为 'YYYY-MM-DD'。
            end (str): 因子生成的结束日期，格式为 'YYYY-MM-DD'。
            name (str): 因子名称，用于指定因子生成代码文件和输出路径。
        """
        self.data = None
        self.time = time
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.name = name

        # 创建日志对象
        self.logger = log_strategy(strategy_name=self.name, factor=True)
        self.logger.info(f"Initialized SignalStrategy for factor '{self.name}'.")

    def generate_signal(self):
        """
        根据因子代码生成指定日期段的因子，并保存到 `./factors/factor_name/{self.time}` 目录中。
        每个日期生成一个以 `YYYY-MM-DD.parquet` 命名的因子文件。
        """
        # 定义因子代码文件路径和输出路径
        factor_code_path = os.path.join('./codes', f'{self.name}.py')
        factor_output_path = os.path.join('./factors', self.name, self.time)

        # 检查因子代码文件是否存在
        if not os.path.exists(factor_code_path):
            self.logger.error(f"Factor code file '{factor_code_path}' does not exist.")
            raise FileNotFoundError(f"Factor code file '{factor_code_path}' does not exist.")

        # 检查因子目录是否存在
        if os.path.exists(factor_output_path):
            user_input = input(
                f"The directory '{factor_output_path}' already exists. "
                f"Do you want to overwrite existing factors? (yes/no): "
            )
            if user_input.lower() not in ["yes", "y"]:
                self.logger.warning(f"Factor generation aborted for '{self.name}' at time '{self.time}'.")
                return
            else:
                self.logger.info(f"Existing factors in '{factor_output_path}' will be overwritten.")
        else:
            # 创建输出目录（如果不存在）
            os.makedirs(factor_output_path)
            self.logger.info(f"Created directory: {factor_output_path}.")

        # 动态加载因子代码模块
        spec = importlib.util.spec_from_file_location(self.name, factor_code_path)
        factor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(factor_module)

        # 检查因子函数是否存在
        if not hasattr(factor_module, 'factor'):
            self.logger.error(f"Factor code file '{factor_code_path}' must contain a 'factor' function.")
            raise AttributeError(f"Factor code file '{factor_code_path}' must contain a 'factor' function.")
        
        # 获取指定日期范围内的交易日列表
        trade_days = get_trade_day_list(self.start, self.end)

        # 因子生成主循环
        for current_date in trade_days:
            self.logger.info(f"Generating factor for date {current_date.strftime('%Y-%m-%d')} at time {self.time}.")

            # 调用因子函数生成因子值
            try:
                factor_values = factor_module.factor(date=current_date, timestamp=self.time)
            except Exception as e:
                self.logger.error(f"Error generating factor for date {current_date.strftime('%Y-%m-%d')} at time {self.time}: {e}")
                continue
            
            # 检查因子值格式
            if not isinstance(factor_values, pd.Series):
                self.logger.error(f"The 'factor' function must return a pd.Series, but got {type(factor_values)}.")
                continue

            # 保存因子值到 Parquet 文件
            factor_file_path = os.path.join(factor_output_path, f"{current_date.strftime('%Y-%m-%d')}.parquet")
            if os.path.exists(factor_file_path):
                self.logger.warning(f"File '{factor_file_path}' already exists. Overwriting.")
            factor_values.to_frame(name=self.name).to_parquet(factor_file_path)

            self.logger.info(f"Factor for {current_date.strftime('%Y-%m-%d')} at time {self.time} saved to {factor_file_path}.")

        self.logger.info(f"Factor generation completed for factor '{self.name}' at time {self.time}.")



class CustomStrategy(ABC):
    """
    自定义多资产交易策略。

    Attributes:
        data (pd.DataFrame): 包含因子和价格数据，用于交易决策。
        model (callable): 预测因子得分的用户模型函数。
        start (str): 策略开始日期。
        end (str): 策略结束日期。
        name (str): 策略名称。
        position (Position): 当前组合持仓。
        equity (float): 当前账户净值，包括现金和资产。
        rebalance (str): 调仓周期，可选值包括 'daily', 'weekly', 'monthly', 'quarterly', '6m', '1y' 等。
        optimizer (str): 选择的优化器类型，例如 "EF", "HRP", "MINVAR" 等。
        target (str): 优化目标，例如 "max_sharpe", "min_volatility" 等。
    """

    REBALANCE_PERIODS = {
        "daily": 1,
        "weekly": 5,
        "monthly": 21,
        "quarterly": 63,
        "6m": 126,
        "1y": 252,
    }


    def __init__(self, data: pd.DataFrame, model, start: str, end: str, name: str, 
                 rebalance: str = "monthly", optimizer: str = "EF", target: str = "max_sharpe"):
        """
        初始化 CustomStrategy 类。

        Parameters:
            data (pd.DataFrame): 包含因子和价格数据的 DataFrame。
            model (callable): 用户提供的模型函数，用于生成预测得分。
            start (str): 策略开始日期。
            end (str): 策略结束日期。
            name (str): 策略名称。
            rebalance (str): 调仓周期，可选值 'daily', 'weekly', 'monthly', 'quarterly', '6m', '1y'。
            optimizer (str): 优化器类型，例如 "EF"（Efficient Frontier），"HRP"（Hierarchical Risk Parity）等。
            target (str): 优化目标，例如 "max_sharpe", "min_volatility"。
        """
        self.data = data
        self.model = model
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.name = name
        self.position =  Position()  # 实例化 Position 类 
        self.orders = []
        self.trades = []
        self.closed_trades = []
        self.equity = 1000000  # 初始资金默认100万
        self.logger = log_strategy(strategy_name=self.name, factor=False)
        self.rebalance = rebalance
        self.optimizer = optimizer
        self.target = target
        self.daily_positions = {}  # 用于记录每日的持仓快照
        self.buy_list = []  # 用于记录需要买入的股票
        self.sell_list = []  # 用于记录需要卖出的股票

        self.logger.info(f"Initialized CustomStrategy '{self.name}' with rebalance period '{self.rebalance}', optimizer '{self.optimizer}', target '{self.target}'.")


    def optimize(self, historical_data: pd.DataFrame):
        """
        使用指定优化器进行组合权重优化。

        Parameters:
            historical_data (pd.DataFrame): 历史价格数据，行索引为日期，列名为股票代码。

        Returns:
            Dict[str, float]: 优化后的组合权重。
        """
        self.logger.info(f"Optimizing portfolio using {self.optimizer} with target {self.target}.")

        # 定义优化器映射
        optimizers = {
            "EF": self.efficient_frontier,
            "HRP": self.hrp,
            "MINVAR": self.min_var,
            "EQUAL": self.equal_weight,  # 新增 Equal Weight 优化器
        }

        if self.optimizer not in optimizers:
            raise ValueError(f"Optimizer '{self.optimizer}' is not supported. Choose from {list(optimizers.keys())}.")

        try:
            optimized_weights = optimizers[self.optimizer](historical_data)
            self.logger.info(f"Final optimized weights after adjustments: {optimized_weights}")
            return optimized_weights
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {}


    def equal_weight(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        使用 Equal Weight 优化器：
        - buy_list 中的股票分配相等正权重。
        - sell_list 中的股票分配相等负权重。

        Parameters:
            historical_data (pd.DataFrame): 历史价格数据（未使用，仅作占位）。

        Returns:
            Dict[str, float]: 优化后的权重。
        """
        if not self.buy_list and not self.sell_list:
            self.logger.warning("Buy and sell lists are empty. Returning empty weights.")
            return {}

        total_buy = len(self.buy_list)
        total_sell = len(self.sell_list)

        weights = {}
        
        # 分配正权重给 buy_list 中的股票
        if total_buy > 0:
            buy_weight = 1 / total_buy
            for symbol in self.buy_list:
                weights[symbol] = buy_weight

        # 分配负权重给 sell_list 中的股票
        if total_sell > 0:
            sell_weight = -1 / total_sell
            for symbol in self.sell_list:
                weights[symbol] = sell_weight

        self.logger.info(f"Equal weight optimization complete. Weights: {weights}")
        return weights


    def efficient_frontier(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        使用 Efficient Frontier 优化器。

        Parameters:
            historical_data (pd.DataFrame): 历史价格数据。

        Returns:
            Dict[str, float]: 优化后的权重。
        """
        mu = expected_returns.mean_historical_return(historical_data)
        S = risk_models.CovarianceShrinkage(historical_data).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        try:
            if self.target == "max_sharpe":
                weights = ef.max_sharpe()
            elif self.target == "min_volatility":
                weights = ef.min_volatility()
            else:
                raise ValueError(f"Target '{self.target}' is not supported for EF.")
            cleaned_weights = ef.clean_weights()
            return self._adjust_weights(cleaned_weights)
        except Exception as e:
            self.logger.error(f"Efficient Frontier optimization failed: {e}")
            return {}

    def hrp(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        使用 Hierarchical Risk Parity (HRP) 优化器。

        Parameters:
            historical_data (pd.DataFrame): 历史价格数据。

        Returns:
            Dict[str, float]: 优化后的权重。
        """
        hrp = HRPOpt(historical_data)
        try:
            weights = hrp.optimize()
            return self._adjust_weights(weights)
        except Exception as e:
            self.logger.error(f"HRP optimization failed: {e}")
            return {}

    def min_var(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        最小化组合波动率优化器。

        Parameters:
            historical_data (pd.DataFrame): 历史价格数据。

        Returns:
            Dict[str, float]: 优化后的权重。
        """
        mu = expected_returns.mean_historical_return(historical_data)
        S = risk_models.sample_cov(historical_data)

        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            return self._adjust_weights(cleaned_weights)
        except Exception as e:
            self.logger.error(f"Minimum variance optimization failed: {e}")
            return {}


    def _adjust_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        调整权重，确保 buy_list 中权重为正且和为 1，sell_list 中权重为负且和为 -1。

        Parameters:
            weights (Dict[str, float]): 原始优化器返回的权重。

        Returns:
            Dict[str, float]: 调整后的权重。
        """
        buy_weights = {k: max(v, 0) for k, v in weights.items() if k in self.buy_list}
        sell_weights = {k: min(v, 0) for k, v in weights.items() if k in self.sell_list}

        # 归一化权重
        if buy_weights:
            buy_sum = sum(buy_weights.values())
            buy_weights = {k: v / buy_sum for k, v in buy_weights.items()}

        if sell_weights:
            sell_sum = abs(sum(sell_weights.values()))
            sell_weights = {k: v / sell_sum for k, v in sell_weights.items()}

        # 合并权重
        adjusted_weights = {**buy_weights, **sell_weights}

        # 确保未在 buy_list 或 sell_list 中的股票权重为 0
        for symbol in weights.keys():
            if symbol not in adjusted_weights:
                adjusted_weights[symbol] = 0

        self.logger.info(f"Adjusted weights: {adjusted_weights}")
        return adjusted_weights


    def is_rebalance_date(self, date: str):
        """
        判断给定日期是否为调仓日。

        Parameters:
            date (str): 当前日期。

        Returns:
            bool: 如果是调仓日返回 True，否则返回 False。
        """
        rebalance_interval = self.REBALANCE_PERIODS[self.rebalance]  # 获取调仓间隔天数
        dates = sorted(self.data["date"].unique())  # 获取所有日期
        date_index = dates.index(date)  # 找到当前日期的索引
        
        # 检查当前索引是否为调仓周期的倍数
        return date_index % rebalance_interval == 0


    def rebalance_portfolio(self, historical_data: pd.DataFrame, date: str):
        """
        根据调仓周期、优化方法和待买入/卖出列表执行组合调仓。
        """
        if not self.is_rebalance_date(date):
            self.daily_positions[date] = self.position.snapshot()
            self.logger.info(f"{date} is not a rebalance date. Snapshot: {self.position.holdings}")
            return

        self.logger.info(f"Rebalancing portfolio on {date}. Buy list: {self.buy_list}, Sell list: {self.sell_list}")
        try:
            target_symbols = list(set(self.buy_list + self.sell_list))
            filtered_data = historical_data[target_symbols]
            filtered_data = filtered_data.dropna(axis=1, how="any")

            if filtered_data.empty:
                self.logger.warning("Filtered data is empty. Skipping rebalance.")
                return

            optimized_weights = self.optimize(filtered_data)

            if not optimized_weights:
                self.logger.warning("No weights returned by optimizer. Skipping rebalance.")
                return

            self.logger.info(f"Optimized weights: {optimized_weights}")

            for symbol, weight in optimized_weights.items():
                self.position.holdings[symbol] = weight #* self.equity
                self.logger.info(f"Updated position for {symbol}: {self.position.holdings[symbol]}")

            self.daily_positions[date] = self.position.snapshot()
            self.buy_list.clear()
            self.sell_list.clear()
        except Exception as e:
            self.logger.error(f"Error during rebalancing on {date}: {e}")


    def buy(self, symbol: str):
        """
        将股票添加到待买入列表中（buy_list）。
        """
        if symbol not in self.buy_list:
            self.buy_list.append(symbol)
            # self.logger.info(f"Added {symbol} to buy list.")
        else:
            self.logger.info(f"{symbol} is already in the buy list.")


    def sell(self, symbol: str):
        """
        将股票添加到待卖出列表中（sell_list）。
        """
        if symbol not in self.sell_list:
            self.sell_list.append(symbol)
            # self.logger.info(f"Added {symbol} to sell list.")
        else:
            self.logger.info(f"{symbol} is already in the sell list.")


    def log_daily_state(self, date):
        """
        记录每天的持仓、订单和交易记录。
        """
        log = {
            "date": date,
            "position": self.position.snapshot(),
            "orders": self.orders.copy(),
            "trades": self.trades.copy(),
            "equity": self.equity,
        }
        self.logger.info(f"Daily log recorded for {date}: {log}")

        
    @abstractmethod
    def init(self):
        """
        初始化策略。

        用户在子类中实现此方法，声明指标并进行预计算。
        """
        pass

    @abstractmethod
    def next(self):
        """
        策略的主运行逻辑。

        用户需在子类中重写此方法，根据调仓周期和预测因子生成交易信号，并更新组合。
        """
        pass