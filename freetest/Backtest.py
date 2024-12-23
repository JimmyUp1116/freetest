import io        
import os
import glob
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
from .utils import get_factor, get_factors, get_trade_day, log_strategy
import quantstats as qs


class SignalTest:
    """
    测试因子类策略，用于因子的回测和分层分析，并生成PDF报告。
    """

    def __init__(self, factors, factor_time, data, benchmark="mean", time="15:00", start=None, end=None, bucket=5, pool="all", rebalance="monthly"):
        self.factors = factors
        self.factor_time = factor_time
        self.data = data
        self.benchmark = benchmark
        self.time = time
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.bucket = bucket
        self.pool = pool
        self.rebalance = rebalance
        self.result = None
        self.stats = None
        self.logger = log_strategy(strategy_name="SignalTest", factor=True)  # 初始化日志对象
        

    def _calculate_indicators(self, returns):
        """
        核心指标计算函数，包括夏普比率、信息比率、波动率等。
        """
        returns = returns.fillna(0)
        annualized_return = qs.stats.cagr(returns)
        cumulative_return = (returns + 1).cumprod().iloc[-1] - 1
        volatility = qs.stats.volatility(returns, annualize=True)
        sharpe_ratio = qs.stats.sharpe(returns)
        max_drawdown = qs.stats.max_drawdown(returns)
        calmar_ratio = qs.stats.calmar(returns)
        # sortino_ratio = qs.stats.sortino_ratio(returns)

        return {
            "annualized_return": annualized_return,
            "cumulative_return": cumulative_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_bucket_stats(self, returns, benchmark):
        """
        计算每个因子分层的收益率和 IR。
        """
        bucket_stats = {}
        for i in range(self.bucket):
            bucket_returns = returns[f"bucket_{i}_return"]
            ir = qs.stats.information_ratio(bucket_returns, benchmark)
            mean_return = bucket_returns.mean()
            bucket_stats[f"bucket_{i}"] = {"IR": ir, "mean_return": mean_return}
        return bucket_stats

    
    def run(self):
        """
        主回测逻辑，生成每日收益率并输出回测结果。
        """
        results = []
        def process_single_date(group):
                """
                处理单个日期的数据，计算分层收益率
                :param group: 某一日期的因子数据
                :return: 每个 bucket 的收益率
                """
                date = group.name
                next_date = get_trade_day(date, 1)

                # 筛选当日和下一交易日的数据
                prices_today = self.data[self.data["date"] == date]
                prices_next = self.data[self.data["date"] == next_date]

                # 合并因子分组数据与当日价格数据
                prices_today = pd.merge(prices_today, group[["symbol", "bucket"]], on="symbol", how="inner")
                prices_next = pd.merge(prices_next, prices_today[["symbol"]], on="symbol", how="inner")

                # 确保 prices_today 和 prices_next 数据完全对齐
                merged_prices = pd.merge(
                    prices_today[["symbol", "close", "bucket"]],
                    prices_next[["symbol", "close"]],
                    on="symbol",
                    how="inner",
                    suffixes=("_today", "_next")
                )

                # 计算收益率 (close_next / close_today - 1)
                merged_prices["returns"] = merged_prices["close_next"] / merged_prices["close_today"] - 1

                # 按 bucket 分组计算平均收益率
                bucket_returns = merged_prices.groupby("bucket")["returns"].mean()

                # 保存结果
                result = {}
                for b in range(self.bucket):
                    result[f"bucket_{b}_return"] = bucket_returns.get(b, float("nan"))
                return pd.Series(result)
        
        for factor in self.factors:
            self.logger.info(f"Backtest initialized for factor: {factor}, from {self.start} to {self.end}")
                             
            # 获取因子数据
            factor_data = get_factor(factor, self.factor_time, self.start, self.end)
            factor_data["bucket"] = pd.qcut(factor_data[factor], self.bucket, labels=False)

            # 使用 apply 处理每个日期分组
            daily_returns_df = factor_data.groupby("date").apply(process_single_date).reset_index()
            daily_returns_df['date'] = pd.to_datetime(daily_returns_df['date'])
            daily_returns_df = daily_returns_df.sort_values(by='date').set_index('date')

            # 计算 1d 和 5d 的累计收益
            cumulative_returns_1d = daily_returns_df.dropna().add(1).cumprod()
            cumulative_returns_5d = (
                daily_returns_df.dropna().rolling(5).apply(lambda x: (x + 1).prod() - 1).add(1).cumprod()
            )

            # 处理基准收益
            if self.benchmark == "mean":
                benchmark_returns = daily_returns_df.mean(axis=1)
            elif isinstance(self.benchmark, pd.Series):
                benchmark_returns = self.benchmark
            else:
                raise ValueError("Unsupported benchmark type. Use 'mean' or a pd.Series.")

            # 对齐基准收益率和每日收益率的时间索引
            benchmark_returns = benchmark_returns.reindex(daily_returns_df.index.unique(), fill_value=0)

            # 计算多空收益和 IR
            long_short_return_1d = (
                daily_returns_df[f"bucket_0_return"] - daily_returns_df[f"bucket_{self.bucket - 1}_return"]
            )
            long_short_return_5d = long_short_return_1d.rolling(5).mean()

            long_short_ir_1d = qs.stats.information_ratio(long_short_return_1d, benchmark=benchmark_returns)
            long_short_ir_5d = qs.stats.information_ratio(long_short_return_5d, benchmark=benchmark_returns)

            # 计算多空策略指标
            long_short_indicators_1d = self._calculate_indicators(long_short_return_1d)
            long_short_indicators_5d = self._calculate_indicators(long_short_return_5d)

            # 调用 _calculate_bucket_stats
            bucket_stats_1d = self._calculate_bucket_stats(daily_returns_df, benchmark=benchmark_returns)
            bucket_stats_5d = self._calculate_bucket_stats(
                daily_returns_df.rolling(5).mean(), benchmark=benchmark_returns
            )

            # 结果记录
            results.append({
                "factor": factor,
                "daily_returns": daily_returns_df,
                "daily_cumulative_returns_1d": cumulative_returns_1d,
                "daily_cumulative_returns_5d": cumulative_returns_5d,
                "long_short_ir_1d": long_short_ir_1d,
                "long_short_ir_5d": long_short_ir_5d,
                **{f"1d_{k}": v for k, v in long_short_indicators_1d.items()},  # 展开 1d 指标
                **{f"5d_{k}": v for k, v in long_short_indicators_5d.items()},  # 展开 5d 指标
                "bucket_stats_1d": bucket_stats_1d,
                "bucket_stats_5d": bucket_stats_5d,
            })

        self.result = results
        return results

    
    def get_stats(self, output_folder="./results/factors/"):
        """
        输出每个因子的统计参数，包括多空策略指标、分层统计等，并存储为 JSON 文件。
        """
        all_stats = []
        for result in self.result:
            factor = result["factor"]

            stats = {
                "factor": factor,
                "long_short_ir_1d": result["long_short_ir_1d"],
                "long_short_ir_5d": result["long_short_ir_5d"],
                **{key: result[key] for key in [
                    "1d_annualized_return", "1d_cumulative_return","1d_volatility",
                    "1d_sharpe_ratio", "1d_max_drawdown", "1d_calmar_ratio",
                    "5d_annualized_return", "5d_cumulative_return", "5d_volatility",
                    "5d_sharpe_ratio", "5d_max_drawdown", "5d_calmar_ratio"
                ]}
            }
            all_stats.append(stats)

            # 保存到 JSON 文件
            factor_output_folder = os.path.join(output_folder, factor)
            if not os.path.exists(factor_output_folder):
                os.makedirs(factor_output_folder)

            output_path = os.path.join(factor_output_folder, f"{factor}.json")
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=4)
        
        return pd.DataFrame(all_stats)


    def plot_signal(self, output_folder="./results/factors/"):
        """
        绘制因子分层累计收益率、直方图、因子分布图，并生成最终的PDF报告。
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for result in self.result:
            factor = result["factor"]
            factor_output_folder = os.path.join(output_folder, factor)
            if not os.path.exists(factor_output_folder):
                os.makedirs(factor_output_folder)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            cumulative_returns_1d = result["daily_cumulative_returns_1d"]
            cumulative_returns_5d = result["daily_cumulative_returns_5d"]
            bucket_stats_1d = result["bucket_stats_1d"]
            bucket_stats_5d = result["bucket_stats_5d"]
            long_short_indicators_1d = result["1d_cumulative_return"]
            long_short_indicators_5d = result["5d_cumulative_return"]
            long_short_ir_1d = result["long_short_ir_1d"]
            long_short_ir_5d = result["long_short_ir_5d"]

            # 添加因子名称
            pdf.add_page()
            pdf.set_font("Times", "B", 14)
            pdf.cell(200, 10, f"Factor: {factor}", ln=True, align="C")
            pdf.set_font("Times", size=12)
            pdf.cell(200, 10, f"Start date: {self.start.date()}   End date: {self.end.date()}", ln=True, align="C")
            pdf.cell(200, 10, f"Rebalance: {self.rebalance}", ln=True, align="C")


            # 创建图表和直方图（并排，保持比例一致）
            # 绘制分层统计表格和因子分布直方图（并排）
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 一行两列

            # 表格
            bucket_table = pd.DataFrame(
                                [
                                    (str(int(bucket.split('_')[1]) + 1),  # 转为整数后转为字符串
                                    "{:.2f}".format(stats["IR"]),  # IR 保留两位小数
                                    "{:.2f} %".format(100 * stats["mean_return"]))  # Mean Return 转为百分比并保留两位小数
                                    for bucket, stats in bucket_stats_1d.items()
                                ],
                                columns=["Bucket", "IR", "Mean Return"]
                            )
            axes[0].axis('tight')
            axes[0].axis('off')
            table = axes[0].table(
                cellText=bucket_table.values,
                colLabels=bucket_table.columns,
                loc='center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)

            # 设置每行高度
            row_heights = 0.1  # 自定义高度，例如 0.25
            for key, cell in table.get_celld().items():
                cell.set_height(row_heights)


            # 设置列宽
            col_widths = [0.2, 0.4, 0.4]  # 对应 Bucket, IR, Mean Return 列宽的比例
            for i, width in enumerate(col_widths):
                for key, cell in table.get_celld().items():
                    if key[1] == i:  # 设置第 i 列的宽度
                        cell.set_width(width)
                    

            # 设置标题加粗
            for (row, col), cell in table.get_celld().items():
                if row == 0:  # 第一行标题
                    cell.set_fontsize(12)
                    cell.set_text_props(weight="bold")

            # 因子分布直方图
            factor_data = get_factor(factor, self.factor_time, self.start, self.end)
            factor_values = factor_data[factor]
            axes[1].hist(factor_values, bins=20, alpha=0.7, color="gray")
            axes[1].set_title(f"{factor} - Distribution")
            axes[1].set_xlabel("Factor Value")
            axes[1].set_ylabel("Frequency")
            axes[1].grid(axis="y", alpha=0.75)

            # 保存并嵌入 PDF
            combined_path = f"{factor_output_folder}/{factor}_table_distribution.png"
            plt.tight_layout()
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            pdf.image(combined_path, x=10, y=None, w=190)  # 确保宽度一致

            # 绘制 1d 和 5d 累计收益率及 IR 图（并排，保持比例一致）
            for horizon, cumulative_returns, cumulative_return, long_short_ir, bucket_stats in [
                ("1d", cumulative_returns_1d, long_short_indicators_1d, long_short_ir_1d, bucket_stats_1d),
                ("5d", cumulative_returns_5d, long_short_indicators_5d, long_short_ir_5d, bucket_stats_5d),
            ]:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 确保比例一致

                # 累计收益率折线图
                for col in cumulative_returns.columns:
                    axes[0].plot(
                        cumulative_returns.index,
                        cumulative_returns[col],
                        label=f"{int(col.split('_')[1]) + 1}",
                        alpha=0.8,
                    )
                axes[0].set_title(f"{factor} - {horizon} Cumulative Returns")
                axes[0].set_xlabel("Date")
                axes[0].set_ylabel("Cumulative Returns")
                axes[0].legend(title="Buckets", loc="upper right", fontsize="small")
                axes[0].text(
                    0.02, 0.95,
                    f"Long-Short Cumulative Return: {cumulative_return:.2%}",
                    transform=axes[0].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )

                # IR 的条形图
                bucket_irs = [stats["IR"] for stats in bucket_stats.values()]
                axes[1].bar(
                    range(1, self.bucket + 1),
                    bucket_irs,
                    alpha=0.6,
                    color="gray",
                    label=f"{horizon} IR",
                )
                axes[1].set_title(f"IR by Bucket ({horizon})")
                axes[1].set_xlabel("Bucket")
                axes[1].set_xticks(range(1, self.bucket + 1))
                axes[1].set_xticklabels(range(1, self.bucket + 1), fontsize=10)
                axes[1].set_ylabel("IR")
                axes[1].legend()
                axes[1].text(
                    0.02, 0.95,
                    f"Long-Short IR: {long_short_ir:.2f}",
                    transform=axes[1].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left'
                )

                plt.tight_layout()

                # 保存图像到本地文件
                chart_path = f"{factor_output_folder}/{factor}_{horizon}_signal_analysis.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                # 添加图像到 PDF 并删除临时文件
                pdf.image(chart_path, x=10, y=None, w=190)  # 确保宽度一致

            # 保存最终 PDF 文件
            pdf_output_path = f"{factor_output_folder}/{factor}.pdf"
            pdf.output(pdf_output_path)
            # 统一删除所有临时图像文件
            for temp_file in glob.glob(f"{factor_output_folder}/*.png"):
                os.remove(temp_file)


    def plot_correlation(self, output_folder="./results/factors/correlation"):
        """
        绘制所有因子的相关性热力图，并保存为 PDF 文件。
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        factor_df = get_factors(self.factors, self.factor_time, self.start, self.end)
        correlation_matrix = factor_df[self.factors].corr()

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            square=True,
        )
        plt.title("Factor Correlation Heatmap")
        plt.tight_layout()

        # 创建 PDF 文件
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(200, 10, "Factor Correlation Heatmap", ln=True, align="C")

        # 将热力图嵌入到 PDF
        chart_path = f"{output_folder}/corr.png"
        plt.savefig(chart_path, format="png", dpi=300)
        pdf.image(chart_path, x=10, y=None, w=180)
        plt.close()

        factor_str = ""
        for factor in self.factors:
            factor_str = factor_str + factor + "_"
        pdf.output(os.path.join(output_folder, f"{factor_str}.pdf"))
        os.remove(chart_path)

        # 返回相关性矩阵
        return correlation_matrix
    


class StrategyTest:
    """
    策略回测类：用于运行策略、生成统计指标。
    """

    def __init__(self, strategy, data, benchmark="mean", start=None, end=None, pool="all", equity=1000000, commission=0.001):
        self.strategy = strategy
        self.data = data
        self.benchmark = benchmark  # 基准：默认为“mean”，即均值
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.result = None  # 每日回测结果
        self.stats = None  # 回测统计指标 
        self.equity = equity  # 初始资金
        self.commission = commission  # 交易成本比例
        self.pool = pool
        self.logger = log_strategy(strategy_name=strategy.name, factor=False)  # 初始化 logger
        self.logger.info(f"Backtest initialized for strategy: {strategy.name}, from {start} to {end}, "
                         f"equity={equity}, commission={commission}.")                                       

    def run(self):
        self.logger.info("Running backtest...")

        filtered_data = self.data[(self.data['date'] >= self.start) & (self.data['date'] <= self.end)]

        if filtered_data.empty:
            self.logger.warning("No data available for the specified date range.")
            return pd.DataFrame()

        # Step 1: 将 `daily_positions` 转为 DataFrame
        position_data = [
            {"date": date, "symbol": symbol, "holdings": holdings}
            for date, snapshot in self.strategy.daily_positions.items()
            for symbol, holdings in snapshot["holdings"].items()
        ]
        position_df = pd.DataFrame(position_data)

        # Step 2: 与价格数据合并
        merged_data = position_df.merge(filtered_data[["date", "symbol", "close", "open"]], on=["date", "symbol"], how="left")
        merged_data["next_date"] = merged_data.groupby("symbol")["date"].shift(-1)
        merged_data["next_close"] = merged_data.groupby("symbol")["close"].shift(-1)
        merged_data["next_open"] = merged_data.groupby("symbol")["open"].shift(-1)

        # Step 3: 计算每只股票的收益率
        merged_data["stock_return"] = (merged_data["next_close"] / merged_data["next_open"]) - 1

        # 计算每日的持仓调整和交易成本
        merged_data["transaction_value"] = abs(merged_data["holdings"] * merged_data["next_open"])
        merged_data["transaction_cost"] = merged_data["transaction_value"] * self.commission

        # 计算每日收益（扣除交易成本）
        merged_data["weighted_return"] = merged_data["holdings"] * merged_data["stock_return"] - merged_data["transaction_cost"] / self.equity

        # 删除无效行
        merged_data = merged_data.dropna(subset=["weighted_return"])

        # Step 4: 汇总每日加权收益率
        daily_returns = merged_data.groupby("date")["weighted_return"].sum()

        # 更新账户净值
        self.equity = self.equity * (1 + daily_returns)

        # Step 5: 生成结果 DataFrame
        cumulative_returns = (1 + daily_returns).cumprod()
        results = pd.DataFrame({
            "daily_returns": daily_returns,
            "cumulative_returns": cumulative_returns
        })
        results.index.name = "date"

        # 保存结果
        self.result = results

        # 日志记录
        self.logger.info(f"Backtest completed. Total dates: {len(results)}")

        return results


    def get_stats(self):
        """
        计算统计指标。
        """
        try:
            self.logger.info("Calculating statistics...")
            stats = {
                "cumulative_return": float(qs.stats.comp(self.result["daily_returns"])),
                "sharpe_ratio": float(qs.stats.sharpe(self.result["daily_returns"])),
                "max_drawdown": float(qs.stats.max_drawdown(self.result["daily_returns"])),
                "volatility": float(qs.stats.volatility(self.result["daily_returns"])),
                "sortino_ratio": float(qs.stats.sortino(self.result["daily_returns"])),
                "final_equity": float(self.equity.iloc[-1]) if isinstance(self.equity, pd.Series) else float(self.equity),
                # "total_commission": float(self.result["daily_returns"].shape[0] * self.commission * self.equity ) # 总交易成本估算
            }

            # 保存统计指标到 JSON 文件
            output_folder = f"./results/strategies/{self.strategy.name}/"
            os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，则创建
            
            # 保存统计指标到 JSON 文件
            json_path = os.path.join(output_folder, f"{self.strategy.name}.json")
            with open(json_path, "w") as f:
                json.dump(stats, f, indent=4)

            self.logger.info(f"Statistics saved to {json_path}.")
            self.stats = stats
            return stats

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            raise


    def plot_strategy(self):
        """
        绘制策略表现图并生成 PDF 报告。
        """
        try:
            self.logger.info("Generating strategy performance plots...")
            output_folder = f"./results/strategies/{self.strategy.name}/"
            os.makedirs(output_folder, exist_ok=True)

            # 设置全局字体为 Linux 可用的字体
            plt.rcParams["font.family"] = "DejaVu Sans"  # 或 "Liberation Sans", "Noto Sans"
            plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

            # 准备数据
            returns = self.result["daily_returns"]#.rename(self.strategy.name).asfreq('B')
            filtered_data = self.data[(self.data['date'] >= self.start) & (self.data['date'] <= self.end)]
            filtered_data = filtered_data.dropna(subset=["close", "open"])

            # 处理基准收益率
            if self.benchmark == "mean":
                filtered_data["next_close"] = filtered_data.groupby("symbol")["close"].shift(-1)
                filtered_data["next_open"] = filtered_data.groupby("symbol")["open"].shift(-1)
                filtered_data["daily_return"] = (filtered_data["next_close"] / filtered_data["next_open"]) - 1
                benchmark_returns = filtered_data.groupby("date")["daily_return"].mean()#.rename("benchmark")
            elif isinstance(self.benchmark, pd.Series):
                benchmark_returns = self.benchmark
            else:
                raise ValueError("Unsupported benchmark type. Use 'mean' or a pd.Series.")
            
            benchmark_returns = benchmark_returns.reindex(self.result.index, fill_value=0)
            # 确保索引是正确的日期格式
            returns.index = pd.DatetimeIndex(returns.index).normalize()
            benchmark_returns.index = pd.DatetimeIndex(benchmark_returns.index).normalize()

            # 对齐频率
            returns = returns.asfreq('B', method='pad')
            benchmark_returns = benchmark_returns.asfreq('B', method='pad')

            yearly_returns = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
            benchmark_yearly_returns = benchmark_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)

            # 初始化 PDF
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=5)
            pdf.add_page()
            pdf.set_font("Times", "B", 14)
            pdf.cell(200, 10, f"Strategy: {self.strategy.name}", ln=True, align="C")
            pdf.set_font("Times", size=12)
            pdf.cell(200, 10, f"Start date: {self.start.date()}   End date: {self.end.date()}", ln=True, align="C")
            # pdf.cell(200, 10, f"Initial Equity: {self.equity}   Commission Rate: {self.commission}", ln=True, align="C")
            pdf.cell(200, 10, f"Rebalance: {self.strategy.rebalance}", ln=True, align="C")

            # 添加统计信息
            pdf.set_font("Times", size=10)
            stats = self.get_stats()
            for key, value in stats.items():
                pdf.cell(200, 5, f"{key}: {value:.2f}", ln=True, align="L")
            
            chart_paths = []

            def save_plot(plot_func, filename, *args, **kwargs):
                save_path = os.path.join(output_folder, filename)
                plot_func(*args, **kwargs, savefig=save_path)
                plt.tight_layout()
                plt.close()
                return save_path

            # 保存所有图表
            chart_paths.append(save_plot(
                qs.plots.returns,
                "returns.png",
                returns,
                benchmark=benchmark_returns,
                cumulative=True
            ))

            # 计算年度收益率
            yearly_returns = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
            benchmark_yearly_returns = benchmark_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)

            # 绘制年度收益率对比图
            plt.figure(figsize=(10, 6))
            bar_width = 0.4
            index = yearly_returns.index.year

            plt.bar(index - 0.2, yearly_returns.values, width=bar_width, label="Strategy", alpha=0.7)
            plt.bar(index + 0.2, benchmark_yearly_returns.values, width=bar_width, label="Benchmark", alpha=0.7)

            plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
            plt.xlabel("Year")
            plt.ylabel("Annual Returns")
            plt.title("Yearly Returns vs Benchmark")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # 保存图表
            save_path = os.path.join(output_folder, "yearly_returns.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            chart_paths.append(save_path)
            chart_paths.append(save_plot(qs.plots.monthly_heatmap, "monthly_heatmap.png", returns))
            chart_paths.append(save_plot(qs.plots.drawdown, "drawdown.png", returns))
            chart_paths.append(save_plot(qs.plots.drawdowns_periods, "drawdowns_periods.png", returns))
            chart_paths.append(save_plot(qs.plots.rolling_volatility, "rolling_volatility.png", returns))
            chart_paths.append(save_plot(qs.plots.rolling_sharpe, "rolling_sharpe.png", returns))
            chart_paths.append(save_plot(qs.plots.rolling_beta, "rolling_beta.png", returns, benchmark=benchmark_returns, ylabel="Beta"))

            # # 添加净值变化图
            # plt.figure(figsize=(10, 6))
            # self.result["equity"] = self.result["cumulative_returns"] * self.equity  # 将累计收益率转为净值
            # plt.plot(self.result.index, self.result["equity"], label="Equity Curve", color="grey")
            # plt.xlabel("Date")
            # plt.ylabel("Equity Value")
            # plt.title("Equity Curve Over Time")
            # plt.legend()
            # plt.grid()
            # equity_curve_path = os.path.join(output_folder, "equity_curve.png")
            # plt.savefig(equity_curve_path, format="png", dpi=300)
            # plt.close()
            # chart_paths.append(equity_curve_path)

            # 将图表添加到 PDF，每页动态布局图表
            image_width = 90
            y_start = pdf.get_y() + 10  # 从统计信息后开始
            for i, path in enumerate(chart_paths):
                if i % 2 == 0 and i != 0:
                    y_start += 70  # 调整下一行图片的起点
                if y_start + 60 > 290:  # 检查是否超过页面高度
                    pdf.add_page()
                    y_start = 20
                x = 10 if i % 2 == 0 else 110  # 左右布局
                pdf.image(path, x=x, y=y_start, w=image_width)

            # 输出 PDF
            pdf_path = os.path.join(output_folder, f"{self.strategy.name}.pdf")
            pdf.output(pdf_path)
            self.logger.info(f"Strategy report saved to {pdf_path}.")

            # 删除临时文件
            for path in chart_paths:
                os.remove(path)

        except Exception as e:
            self.logger.error(f"Failed to generate strategy performance plots: {e}")
            raise
