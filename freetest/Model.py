import os
import pickle
import json
import joblib
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
)


class Model:
    """
    用于将因子数据集转换为预测值的类，可以支持线性回归、逻辑回归、自定义模型等多种结构。
    """

    def __init__(self, data: pd.DataFrame = None, name: str = "default_model", save_model: bool = True):
        """
        初始化 Model 类。

        参数:
            data (pd.DataFrame): 用于预测的数据集。
            name (str): 模型名称，用于保存和加载模型。
            save_model (bool): 是否保存模型到文件，默认为 True。
        """
        self.data = data
        self.prediction = None
        self.name = name
        self.save_model = save_model

        # 确保模型目录存在
        if not os.path.exists("./models"):
            os.makedirs("./models")

    def linear_fit(self, target: str, l1: float = 0.0, l2: float = 0.0) -> pd.Series:
        """
        构建线性回归模型，可设置 L1 和 L2 正则化系数。

        参数:
            target (str): 数据集中目标列的名称。
            l1 (float): L1 正则化系数，默认为 0。
            l2 (float): L2 正则化系数，默认为 0。

        返回:
            pd.Series: 预测值，索引为股票代码。
        """

        # 分离特征和目标变量
        X = self.data.drop(columns=[target])
        y = self.data[target]

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 选择正则化类型
        if l1 > 0 and l2 == 0:
            model = Lasso(alpha=l1)  # L1 正则化
        elif l2 > 0 and l1 == 0:
            model = Ridge(alpha=l2)  # L2 正则化
        else:
            model = LinearRegression()  # 无正则化

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 预测测试集并计算误差
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Linear Model MSE: {mse:.4f}")

        # 保存模型
        if self.save_model:
            model_path = f"./models/{self.name}_linear.pkl"
            joblib.dump(model, model_path)
            print(f"Linear model saved at: {model_path}")

        # 预测所有数据
        predictions = model.predict(X_scaled)

        return pd.Series(predictions, index=self.data.index, name="linear_prediction")

    def log_fit(self, target: str, l1: float = 0.0, l2: float = 0.0) -> pd.Series:
        """
        构建逻辑回归模型，可设置 L1 和 L2 正则化系数。

        参数:
            target (str): 数据集中目标列的名称。
            l1 (float): L1 正则化系数，默认为 0。
            l2 (float): L2 正则化系数，默认为 0。

        返回:
            pd.Series: 预测值，索引为股票代码。
        """
        # 分离特征和目标变量
        X = self.data.drop(columns=[target])
        y = self.data[target]

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 选择正则化类型
        penalty = None
        if l1 > 0 and l2 == 0:
            penalty = "l1"  # L1 正则化
            solver = "liblinear"  # 支持 L1 的 solver
        elif l2 > 0 and l1 == 0:
            penalty = "l2"  # L2 正则化
            solver = "lbfgs"  # 默认 solver
        else:
            penalty = None
            solver = "lbfgs"

        model = LogisticRegression(penalty=penalty, C=1/(l1 + l2) if (l1 > 0 or l2 > 0) else 1.0, solver=solver, max_iter=1000)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 预测测试集并计算准确率
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Logistic Model Accuracy: {accuracy:.4f}")

        # 保存模型
        if self.save_model:
            model_path = f"./models/{self.name}_logistic.pkl"
            joblib.dump(model, model_path)
            print(f"Logistic model saved at: {model_path}")

        # 预测概率（取正类概率）
        probabilities = model.predict_proba(X_scaled)[:, 1]

        return pd.Series(probabilities, index=self.data.index, name="logistic_prediction")

    def fit_with_function(self, model_function: callable, target: str, var: list) -> pd.Series:
        """
        使用用户传入的模型函数对数据进行训练和预测。

        参数:
            model_function (callable): 用户定义的函数，用于创建和训练模型。必须返回 (model, predictions)。
            target (str): 数据集中目标列的名称。

        返回:
            pd.Series: 预测值，索引为股票代码。
        """
        if self.data is None:
            raise ValueError("Data cannot be None.")
        
        if target not in self.data.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")

        # 提取特征和目标
        X = self.data[var]
        y = self.data[target]

        # 调用用户定义的模型函数
        model, predictions = model_function(X, y)

        if not isinstance(predictions, pd.Series):
            raise ValueError("The predictions returned by the model function must be a pd.Series.")

        # 保存模型（如果需要）
        if self.save_model:
            model_path = os.path.join("./models", f"{self.name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # 保存预测结果
        self.prediction = predictions
        return predictions

    def load_model(self, model_name) -> object:
        """
        加载保存的模型。

        返回:
            object: 加载的模型对象。
        """
        model_path = os.path.join("./models", f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def predict_with_model(self, data: pd.DataFrame, var: list) -> pd.Series:
        
        """
        使用加载的模型生成预测值。

        参数:
            data (pd.DataFrame): 需要用来预测的数据
            var (list): 用于预测的自变量名称列表

        返回:
            pd.Series: 预测值。
        """
        model = self.load_model(self.name)
        if data is None:
            raise ValueError("Data cannot be None.")
        
        X = data[var]
        predictions = pd.Series(model.predict(X), index=X.index)
        self.prediction = predictions
        return predictions
    

    def calculate_metrics(self, data: pd.DataFrame, target: str, pred: str, save_name: str):
        """
        计算预测值的评估指标，包括 R^2、MSE、MAE、Accuracy、F1 Score 等，并保存到 JSON 文件。

        参数:
            data (pd.DataFrame): 包含真实值和预测值的数据集。
            target (str): 真实值列的名称。
            pred (str): 预测值列的名称。

        返回:
            dict: 包含各项评估指标的字典。
        """
        # 确保目标列和预测值列存在
        if target not in data.columns or pred not in data.columns:
            raise ValueError(f"Both target '{target}' and prediction '{pred}' columns must exist in the dataset.")
        
        # 提取真实值和预测值
        y_true = data[target]
        y_pred = data[pred]

        # 判断任务类型（回归或分类）
        is_classification = len(np.unique(y_true)) < 20  # 简单判断类别数量，少于20视为分类任务
        
        # 计算回归指标
        metrics = {
            "R2_Score": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
        }

        # 如果是分类任务，计算 Accuracy 和 F1 Score
        if is_classification:
            y_true_binary = np.round(y_true)  # 将真实值转换为二进制类别
            y_pred_binary = np.round(y_pred)  # 将预测值转换为二进制类别
            metrics["Accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
            metrics["F1_Score"] = f1_score(y_true_binary, y_pred_binary)

        # 保存结果到 JSON 文件
        output_folder = os.path.join("./results/models", self.name)
        os.makedirs(output_folder, exist_ok=True)  # 确保目录存在
        output_path = os.path.join(output_folder, f"{self.name}_{save_name}.json")

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {output_path}")
        return metrics