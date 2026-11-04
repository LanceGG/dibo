# 克隆自聚宽文章：https://www.joinquant.com/post/64178
# 标题：多策略11：去伪存真，拥抱不择时的核心逻辑
# 作者：O_iX

# 导入函数库
from jqdata import *
from jqfactor import get_factor_values
import datetime
import math
from scipy.optimize import minimize
import numpy as np
import pandas as pd


# -------------------- 运行调度函数 --------------------
def initialize(context):

    set_benchmark("000300.XSHG")  # 设定沪深300作为基准
    set_option("avoid_future_data", True)  # 打开防未来函数
    set_option("use_real_price", True)  # 开启动态复权模式(真实价格)
    log.info("初始函数开始运行且全局只运行一次")  # 输出内容到日志 log.info()
    log.set_level("order", "error")  # 过滤掉order系列API产生的比error级别低的log
    set_slippage(PriceRelatedSlippage(0.006), type="stock")

    # 全局变量
    g.strategys = {}
    # g.portfolio_value_proportion = [1, 0, 0, 0]  # 测试版
    g.portfolio_value_proportion = [0.3, 0.5, 0.1, 0.1]  # 标准版
    g.positions = {i: {} for i in range(len(g.portfolio_value_proportion))}  # 记录每个子策略的持仓股票

    # 子策略执行计划
    if g.portfolio_value_proportion[0] > 0:
        run_weekly(jsg_adjust, 1, "11:00")
        run_daily(jsg_check, "14:55")
    if g.portfolio_value_proportion[1] > 0:
        run_weekly(all_day_adjust, 1, "11:01")
    if g.portfolio_value_proportion[2] > 0:
        run_monthly(simple_roa_adjust, 1, "11:02")
        run_daily(simple_roa_check, "14:55")
    if g.portfolio_value_proportion[3] > 0:
        run_daily(etf_rotation_adjust, "11:03")
    # 尾盘交易
    run_daily(end_trade, "14:59")
    # 盘后分析
    run_daily(analysis, "14:59")


def process_initialize(context):
    g.strategys = {
        name: cls(context, index=idx, name=name, display_name=display_name)
        for display_name, name, cls, idx in [
            ("搅屎棍策略", "jsg_strategy", JSG_Strategy, 0),
            ("全天候策略", "all_day_stragety", All_Day_Strategy, 1),
            ("简单ROA策略", "simple_roa_stragegy", Simple_ROA_Strategy, 2),
            ("核心资产轮动策略", "etf_rotation_strategy", Etf_Rotation_Strategy, 3),
        ]
    }


# 事件处理
def on_event(context, event):
    # 处理分红事件
    if event.name == "Dividends":
        for d in event.dividends:
            scale = d.get("scale_factor", 1)
            if scale != 1:
                print(f"收到送股事件: {event.security}:{event.dividends}")
                code = event.security.code
                for pid, pos in g.positions.items():
                    if code in pos:
                        old = pos[code]
                        pos[code] = round(old * scale)
                        print(f"调整持仓: 组合 {pid}, 股票 {code}, 数量 {old} -> {pos[code]}")


# 尾盘处理
def end_trade(context):

    marked = {s for d in g.positions.values() for s in d}
    for stock in context.portfolio.positions:
        if stock not in marked or context.portfolio.positions[stock].total_amount < 100:
            order_target_value(stock, 0)
            log.info(f"卖出{stock}因送股未记录在持仓中")
            for pid in list(g.positions.keys()):
                g.positions[pid].pop(stock, None)


# 盘后分析
def analysis(context):
    current_data = get_current_data()
    positions = context.portfolio.positions

    paused = [current_data[s].name for s in positions if current_data[s].paused]
    others = [s for s in positions if not current_data[s].paused]

    st = [current_data[s].name for s in others if current_data[s].is_st]
    limitup = [current_data[s].name for s in others if current_data[s].last_price == current_data[s].high_limit]
    limitdown = [current_data[s].name for s in others if current_data[s].last_price == current_data[s].low_limit]

    parts = []
    for label, lst in (
        ("ST", st),
        ("停牌", paused),
        ("涨停⬆", limitup),
        ("跌停⬇", limitdown),
    ):
        if lst:
            parts.append(f"{label}: {lst}")
    if parts:
        log.info("; ".join(parts))


def jsg_adjust(context):
    g.strategys["jsg_strategy"].adjust()


def jsg_check(context):
    g.strategys["jsg_strategy"].check()


def all_day_adjust(context):
    g.strategys["all_day_stragety"].adjust()


def simple_roa_adjust(context):
    g.strategys["simple_roa_stragegy"].adjust()


def simple_roa_check(context):
    g.strategys["simple_roa_stragegy"].check()


def etf_rotation_adjust(context):
    g.strategys["etf_rotation_strategy"].adjust()


# -------------------- 策略基类 --------------------
class Strategy:

    def __init__(self, context, index, name, display_name):
        self.context = context
        self.index = index
        self.name = name
        self.display_name = display_name
        self.stock_sum = 1
        self.buffer_sum = 1
        self.hold_list = []
        self.min_money = 20000  # 最小交易额(限制手续费)

    # 获取持仓市值
    def get_total_value(self):
        if not g.positions[self.index]:
            return 0
        return sum(self.context.portfolio.positions[key].price * value for key, value in g.positions[self.index].items())

    # 调仓(targets为字典，key为股票代码，value为目标市值)
    def _adjust(self, targets):

        current_data = get_current_data()

        # 获取已持有列表
        self.hold_list = list(g.positions[self.index].keys())
        fixed = self.get_untradeable_stock(self.hold_list)
        portfolio = self.context.portfolio

        # 获取目标策略市值
        target_value = self.context.portfolio.total_value * g.portfolio_value_proportion[self.index]

        # 清仓被调出的
        for stock in self.hold_list:
            if stock not in targets:
                self.order_target_value_(stock, 0)

        # 先卖出
        for stock, weight in targets.items():
            if stock in fixed:
                continue
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions[self.index].get(stock, 0) * price
            if value - target > max(self.min_money, price * 100):
                self.order_target_value_(stock, target)

        # 后买入
        for stock, weight in targets.items():
            if stock in fixed:
                continue
            target = target_value * weight
            price = current_data[stock].last_price
            value = g.positions[self.index].get(stock, 0) * price
            if min(target - value, portfolio.available_cash) > max(self.min_money, price * 100):
                self.order_target_value_(stock, target)

    # 自定义下单(涨跌停不交易)
    def order_target_value_(self, security, value):
        current_data = get_current_data()

        if current_data[security].is_st:
            log.info(f"{security}: ST股票")

        # 检查标的是否停牌、涨停、跌停
        if current_data[security].paused:
            log.info(f"{security}: 今日停牌")
            return False

        if current_data[security].last_price == current_data[security].high_limit:
            log.info(f"{security}: 当前涨停")
            return False

        if current_data[security].last_price == current_data[security].low_limit:
            log.info(f"{security}: 当前跌停")
            return False

        # 获取当前标的的价格
        price = current_data[security].last_price

        # 获取当前策略的持仓数量
        current_position = g.positions[self.index].get(security, 0)

        # 计算目标持仓数量
        target_position = (int(value / price) // 100) * 100 if price != 0 else 0

        # 计算需要调整的数量
        adjustment = target_position - current_position

        # 检查是否当天买入卖出
        closeable_amount = self.context.portfolio.positions[security].closeable_amount if security in self.context.portfolio.positions else 0
        if adjustment < 0 and closeable_amount == 0:
            log.info(f"{security}: 当天买入不可卖出")
            return False

        # 下单并更新持仓
        if adjustment != 0:
            o = order(security, adjustment)
            if o:
                # 更新持仓数量
                filled = o.filled if o.is_buy else -o.filled
                g.positions[self.index][security] = filled + current_position
                # 如果目标持仓为零，移除该证券
                if g.positions[self.index][security] == 0:
                    g.positions[self.index].pop(security, None)
                # 更新持有列表
                self.hold_list = list(g.positions[self.index].keys())
                return True
        return False

    # 基础过滤(过滤科创北交、ST、停牌、次新股/日)
    def filter_basic_stock(self, stock_list, include_kcbj=False, days=360):
        """默认过滤ST停牌、科创北交、次新股(上市不满360天)"""
        current_data = get_current_data()

        stock_list = stock_list if include_kcbj else [stock for stock in stock_list if not (stock[0] == "4" or stock[0] == "8" or stock[:2] == "68")]

        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
            and not current_data[stock].is_st
            and "ST" not in current_data[stock].name
            and "*" not in current_data[stock].name
            and "退" not in current_data[stock].name
            and not self.context.previous_date - get_security_info(stock).start_date < datetime.timedelta(days)
        ]

    # 过滤近几日涨停过的股票
    def filter_limitup_stock(self, stock_list, days):
        df = get_price(
            stock_list,
            end_date=self.context.previous_date,
            frequency="daily",
            fields=["close", "high_limit"],
            count=days,
            panel=False,
        )
        df = df[df["close"] == df["high_limit"]]
        filterd_stocks = df.code.drop_duplicates().tolist()
        return [stock for stock in stock_list if stock not in filterd_stocks]

    # 过滤重复行业，每个行业允许持有的最大股票数量为n
    def filter_industry(self, stocks, n):
        industry_map = {s: info.get("sw_l1", {}).get("industry_name", "") for s, info in get_industry(stocks).items()}
        counts = {}
        result = []
        for s in stocks:
            ind = industry_map.get(s, "")
            if counts.get(ind, 0) < n:
                result.append(s)
                counts[ind] = counts.get(ind, 0) + 1

        return result

    # 检查持仓中曾经涨停但当前未涨停的股票与ST股票
    def _check(self):
        hold = list(g.positions[self.index].keys())
        if not hold:
            return []

        current_data = get_current_data()

        limitup = set(hold) - set(self.filter_limitup_stock(hold, 3))
        fixed = set(self.get_untradeable_stock(hold))
        return [s for s in hold if (s in limitup or current_data[s].is_st) and s not in fixed]

    # 识别无法交易的股票（停牌、涨跌停）
    def get_untradeable_stock(self, stocks):
        current_data = get_current_data()
        return [
            stock
            for stock in stocks
            if current_data[stock].paused or current_data[stock].last_price in (current_data[stock].high_limit, current_data[stock].low_limit)
        ]

    # 根据调仓逻辑计算最终保留的股票列表
    def get_adjusted_stocks(self, selected, sell):
        positions = list(g.positions[self.index].keys())
        # 先保留无法交易的股票
        fixed = self.get_untradeable_stock(positions)
        # 从已选列表中去掉 fixed 和 将要卖出的股票，作为候选
        candidates = [s for s in selected if s not in fixed and s not in sell]
        # 再保留缓冲池中的股票（按候选顺序）
        buffered = [s for s in positions if s in candidates[: self.buffer_sum]]
        # 计算还需保留的名额
        remaining = self.stock_sum - len(fixed) - len(buffered)
        # 从候选中去掉已缓冲的，补足剩余名额
        rest = [s for s in candidates if s not in buffered][:remaining]
        # 按顺序合并并去重返回最终名单
        return list(dict.fromkeys(fixed + buffered + rest))

    # 排除金融行业股票
    def filter_financial_stocks(self, stocks):
        """过滤掉金融行业的股票"""
        financial_industries = ["银行I", "非银金融I"]
        industry_data = get_industry(stocks)
        return [stock for stock in stocks if not industry_data[stock].get("sw_l1", {}).get("industry_name", "") in financial_industries]

    # 剔除审计意见异常的股票
    def filter_audit_opinion_stocks(self, stocks, years):
        a = finance.STK_AUDIT_OPINION
        df = finance.run_query(
            query(
                a.code,
            )
            .filter(
                a.code.in_(stocks),
                a.pub_date >= self.context.previous_date - datetime.timedelta(365 * years),
                a.pub_date <= self.context.previous_date,
                a.opinion_type_id != 6,
                a.opinion_type_id != 1,
            )
            .order_by(a.pub_date.desc())
        )
        excluded_stocks = df["code"].drop_duplicates().tolist()
        return [stock for stock in stocks if stock not in excluded_stocks]


# -------------------- 各具体策略类 --------------------


# 搅屎棍策略
class JSG_Strategy(Strategy):

    def __init__(self, context, index, name, display_name):
        super().__init__(context, index, name, display_name)

        self.stock_sum = 6
        self.buffer_sum = 9  # 缓冲池大小,数值取值在 stock_sum ~ stock_sum*2 之间较为合适

    # 过滤股票
    def filter(self):
        stocks = get_index_stocks("399101.XSHE")
        # stocks = get_all_securities(["stock"]).index.tolist()
        stocks = self.filter_basic_stock(stocks)
        stocks = self.filter_limitup_stock(stocks, 5)
        df = get_fundamentals(
            query(
                valuation.code,
            )
            .filter(
                valuation.code.in_(stocks),
                indicator.adjusted_profit > 0,
                valuation.pb_ratio > 0,
            )
            .order_by(valuation.market_cap.asc())
        ).head(200)
        # print(df.head(20))
        stocks = df.code.tolist()
        stocks = self.filter_audit_opinion_stocks(stocks, 2)

        return stocks

    # 择时
    def select(self):
        return self.filter()[: self.stock_sum * 2]

    # 调仓
    def adjust(self):
        stocks = self.get_adjusted_stocks(self.select(), [])
        self._adjust({stock: round(1 / len(stocks), 3) for stock in stocks})

    # 卖出非连板股票
    def check(self):
        banner = self._check()
        if banner:
            stocks = self.get_adjusted_stocks(self.select(), banner)
            self._adjust({s: round(1 / len(stocks), 3) for s in stocks})


# 全天候ETF策略
class All_Day_Strategy(Strategy):

    def __init__(self, context, index, name, display_name):
        super().__init__(context, index, name, display_name)

        # 全天候ETF组合参数
        self.etf_pool = [
            "511010.XSHG",  # 国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指100
            "512890.XSHG",  # 红利低波
        ]
        # 标的仓位占比
        self.rates = [0.3, 0.2, 0.2, 0.3]

    # 调仓
    def adjust(self):
        self._adjust({etf: round(rate, 3) for etf, rate in zip(self.etf_pool, self.rates)})


# 简单ROA策略
class Simple_ROA_Strategy(Strategy):
    def __init__(self, context, index, name, display_name):
        super().__init__(context, index, name, display_name)

        self.stock_sum = 1

    # 选股
    def filter(self):
        stocks = get_all_securities("stock", date=self.context.previous_date).index.tolist()
        stocks = self.filter_basic_stock(stocks)
        stocks = self.filter_limitup_stock(stocks, 5)
        df = (
            get_fundamentals(
                query(valuation.code, valuation.pb_ratio, indicator.roa, indicator.statDate).filter(
                    valuation.code.in_(stocks),
                    valuation.pb_ratio > 0,
                    valuation.pb_ratio < 1,
                    indicator.adjusted_profit > 0,
                )
            )
            .sort_values(by="roa", ascending=False)
            .head(10)
        )
        # print(df)
        stocks = df.code.tolist()
        stocks = self.filter_audit_opinion_stocks(stocks, 2)
        if not stocks:
            return []
        return stocks

    # 择时
    def select(self):
        return self.filter()[: self.stock_sum * 2]

    # 调仓
    def adjust(self):
        stocks = self.get_adjusted_stocks(self.select(), [])
        self._adjust({stock: round(1 / len(stocks), 3) for stock in stocks})

    # 检查昨日涨停票
    def check(self):
        banner = self._check()
        if banner:
            stocks = self.get_adjusted_stocks(self.select(), banner)
            self._adjust({s: round(1 / len(stocks), 3) for s in stocks})


# 核心资产轮动策略
class Etf_Rotation_Strategy(Strategy):
    def __init__(self, context, index, name, display_name):
        super().__init__(context, index, name, display_name)

        self.etf_pool = [
            # 境外
            "513100.XSHG",  # 纳指ETF
            "513520.XSHG",  # 日经ETF
            "513030.XSHG",  # 德国ETF
            # 商品
            "518880.XSHG",  # 黄金ETF
            "159980.XSHE",  # 有色ETF
            "159985.XSHE",  # 豆粕ETF
            "501018.XSHG",  # 南方原油
            # 债券
            "511090.XSHG",  # 30年国债ETF
            # 国内
            "513130.XSHG",  # 恒生科技
        ]

    # 选取ETF
    def filter(self, name, days, max_score):
        data = pd.DataFrame(index=self.etf_pool, columns=["annualized_returns", "r2", "score"])
        current_data = get_current_data()
        for etf in self.etf_pool:
            # 获取数据
            df = attribute_history(etf, days, "1d", ["close", "high"])
            prices = np.append(df["close"].values, current_data[etf].last_price)

            # 设置参数
            y = np.log(prices)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))

            # 计算年化收益率
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1

            # 计算R²
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0

            # 计算得分
            data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]

            # 过滤近3日跌幅超过5%的ETF
            if min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
                data.loc[etf, "score"] = 0

        # 保留两位小数
        data[["annualized_returns", "r2", "score"]] = data[["annualized_returns", "r2", "score"]].apply(pd.to_numeric, errors="coerce").round(2)

        # 打印结果
        # log.info(f"ETF轮动策略-{name}:")
        # log.info(data[(data["score"] >= 0)].sort_values(by="score", ascending=False).head(5))

        # 过滤ETF，并按得分降序排列
        data = data[(data["score"] > 0) & (data["score"] < max_score)].sort_values(by="score", ascending=False)

        if data.empty:
            return "511880.XSHG"

        return data.index.tolist()[0]

    # 选股
    def select(self):
        # 短周期动量选股
        etf1 = self.filter("短周期动量", 25, 6)
        # 长周期动量选股
        etf2 = self.filter("长周期动量", 250, 0.5)
        return [etf1, etf2] if etf1 != etf2 else [etf1]

    # 调仓
    def adjust(self):
        targets = self.select()
        self._adjust({etf: round(1 / len(targets), 3) for etf in targets})
