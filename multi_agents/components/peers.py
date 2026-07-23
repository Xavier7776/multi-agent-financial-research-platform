"""同行解析器（PeerResolver）— 统一同行数据获取入口。

架构（按优先级链降级）：
    Priority 1: FMPStockPeersSource  — FMP /stock-peers API（动态，全覆盖美股）
    Priority 2: YAMLIndustrySource   — config/industry_peers.yaml（静态，零成本，含 A 股）
    Priority 3: LLMPeerSource        — LLM 生成 2-3 家同行（兜底，有成本）

每个 source 返回 PeerResult。PeerResolver 按序调用，首个成功即返回。
FMP API 连续失败触发熔断，避免雪崩。所有结果可由 FinancialDataTool._cache 缓存。

使用方式（一般不直接用，由 FinancialDataTool / XueqiuDataTool 内部调用）：
    resolver = PeerResolver(task=task, fmp_fetcher=fmp_get)
    peers = await resolver.resolve("AAPL", overview={...})
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def _call_model(prompt, model=None, **kwargs):
    """延迟 import call_model，避免循环依赖 + 便于单元测试 mock。"""
    from multi_agents.agents.utils.llms import call_model
    return call_model(prompt, model, **kwargs)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class PeerInfo:
    """统一的同行数据结构（与原 dict 字段对齐：ticker/name/pe/pb/roe/revenue_growth/market_cap）。"""
    ticker: str
    name: str = ""
    market_cap: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None

    def to_dict(self) -> dict:
        """转换为下游期望的 dict 格式（向后兼容 editor/writer/reviewer）。"""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "market_cap": self.market_cap,
            "pe": self.pe,
            "pb": self.pb,
            "roe": self.roe,
            "revenue_growth": self.revenue_growth,
        }


@dataclass
class PeerResult:
    """数据源返回结果。"""
    peers: list[PeerInfo] = field(default_factory=list)
    source: str = ""
    success: bool = True
    error: Optional[str] = None


# ======================================================================
# 熔断器（防 FMP API 故障雪崩）
# ======================================================================

class CircuitBreaker:
    """简单熔断器。

    状态转换：
        CLOSED ──(fail_count >= threshold)──→ OPEN
        OPEN   ──(cooldown elapsed)───────→ HALF_OPEN
        HALF_OPEN ──(success)────────────→ CLOSED
        HALF_OPEN ──(failure)────────────→ OPEN
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._fail_count = 0
        self._last_fail_time: float = 0
        self._state = "CLOSED"

    def allow(self) -> bool:
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN":
            if time.time() - self._last_fail_time >= self._cooldown:
                self._state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN

    def record_success(self):
        self._fail_count = 0
        self._state = "CLOSED"

    def record_failure(self):
        self._fail_count += 1
        self._last_fail_time = time.time()
        if self._fail_count >= self._threshold:
            self._state = "OPEN"


# ======================================================================
# 数据源
# ======================================================================

class AbstractPeerSource:
    """同行数据源抽象基类。所有 source 实现同一接口，便于测试 mock。"""

    async def fetch(self, ticker: str, **context) -> PeerResult:
        raise NotImplementedError


class FMPStockPeersSource(AbstractPeerSource):
    """FMP /stable/stock-peers API 数据源。

    覆盖 FMP 数据库内全部美股，单次调用返回 symbol + companyName + marketCap。
    拿到 peer ticker 后仍需通过 fetch_peer_info 补查 PE/PB/ROE。

    免费版限制：250 次/天，与财报请求共享配额，故加熔断器。
    """

    ENDPOINT = "stock-peers"
    MAX_PEERS = 6

    def __init__(self, fmp_fetcher, circuit_breaker: CircuitBreaker,
                 peer_info_fetcher=None):
        """
        Args:
            fmp_fetcher: 同步 FMP GET 函数（如 _fmp_get）
            circuit_breaker: FMP API 熔断器
            peer_info_fetcher: 异步回调 (ticker:str) -> PeerInfo，用于补查 PE/PB/ROE。
                               若为 None，则只返回 ticker+name+market_cap。
        """
        self._fetch = fmp_fetcher
        self._breaker = circuit_breaker
        self._peer_info_fetcher = peer_info_fetcher

    async def fetch(self, ticker: str, **context) -> PeerResult:
        # A 股/港股数字代码不适用 FMP stock-peers（仅美股）
        if ticker.isdigit():
            return PeerResult(success=False, source="fmp_api",
                              error="ashare_ticker_skipped")

        if not self._breaker.allow():
            return PeerResult(success=False, source="fmp_api",
                              error="circuit_breaker_open")

        try:
            raw = await asyncio.to_thread(
                self._fetch, self.ENDPOINT, {"symbol": ticker}
            )
            if not raw or not isinstance(raw, list):
                self._breaker.record_failure()
                return PeerResult(success=False, source="fmp_api",
                                  error="empty_response")

            self._breaker.record_success()
            base_peers = []
            for p in raw[: self.MAX_PEERS]:
                symbol = p.get("symbol", "")
                if not symbol or symbol.upper() == ticker.upper():
                    continue
                base_peers.append(PeerInfo(
                    ticker=symbol,
                    name=p.get("companyName", ""),
                    market_cap=p.get("mktCap"),
                ))

            if not base_peers:
                return PeerResult(success=False, source="fmp_api",
                                  error="no_peers_returned")

            # 若有 peer_info_fetcher，补查 PE/PB/ROE
            if self._peer_info_fetcher:
                tasks = [self._peer_info_fetcher(p.ticker) for p in base_peers]
                enriched = await asyncio.gather(*tasks, return_exceptions=True)
                final = []
                for base, info in zip(base_peers, enriched):
                    if isinstance(info, PeerInfo) and info.name:
                        final.append(info)
                    else:
                        final.append(base)  # 退化为只含 ticker+name+market_cap
                base_peers = final

            return PeerResult(peers=base_peers, source="fmp_api")

        except Exception as e:
            self._breaker.record_failure()
            return PeerResult(success=False, source="fmp_api", error=str(e))


class YAMLIndustrySource(AbstractPeerSource):
    """YAML 静态配置数据源（零成本降级，含 A 股 + 美股 19 行业）。

    配置文件路径：config/industry_peers.yaml
    启动时加载一次（配置不常改，不做热加载以保持简单）。
    """

    _data: dict[str, list[str]] = {}
    _loaded: bool = False

    def __init__(self, config_path: str = None):
        if not YAMLIndustrySource._loaded:
            self._load(config_path)

    @classmethod
    def _load(cls, config_path: str = None):
        path = Path(config_path) if config_path else \
            Path(__file__).resolve().parents[2] / "config" / "industry_peers.yaml"
        try:
            with open(path, encoding="utf-8") as f:
                cls._data = yaml.safe_load(f) or {}
            cls._loaded = True
            logger.info(f"[YAMLIndustrySource] 加载 {len(cls._data)} 个行业映射: {path}")
        except Exception as e:
            logger.warning(f"[YAMLIndustrySource] 加载失败 {path}: {e}")
            cls._data = {}
            cls._loaded = True

    async def fetch(self, ticker: str, **context) -> PeerResult:
        industry = context.get("industry", "")
        ticker_upper = ticker.upper()

        # Priority 2a: 按行业名匹配
        if industry and industry in self._data:
            peers = [p for p in self._data[industry]
                     if str(p).upper() != ticker_upper]
            if peers:
                return PeerResult(
                    peers=[PeerInfo(ticker=str(p)) for p in peers[:6]],
                    source="yaml_config"
                )

        # Priority 2b: 反向查找（ticker 本身出现在哪个行业列表）
        for ind_peers in self._data.values():
            if ticker_upper in [str(p).upper() for p in ind_peers]:
                peers = [p for p in ind_peers if str(p).upper() != ticker_upper]
                if peers:
                    return PeerResult(
                        peers=[PeerInfo(ticker=str(p)) for p in peers[:6]],
                        source="yaml_config"
                    )

        return PeerResult(success=False, source="yaml_config", error="no_match")


class LLMPeerSource(AbstractPeerSource):
    """LLM 生成同行数据源（兜底，A 股 + 美股通用）。

    策略：
    1. LLM 根据公司名 + 行业生成 2-3 个 peer ticker
    2. 逐个用对应数据源（A 股→雪球，美股→FMP）验证并补查 PE/PB/ROE
    3. 成本 ~$0.001/次，仅在 FMP API + YAML 都失败时触发
    """

    def __init__(self, task: dict):
        self._task = task or {}

    async def fetch(self, ticker: str, **context) -> PeerResult:
        overview = context.get("overview", {})
        name = overview.get("name", "") or ticker
        sector = overview.get("sector", "")
        industry = overview.get("industry", "")

        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个金融行业分析师。根据公司信息，列出 2-3 家同行业可比公司"
                    "的股票代码。只返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"公司：{name}（{ticker}）\n"
                    f"行业：{sector} / {industry}\n"
                    f"请返回 2-3 家同行业对手的股票代码。"
                    f"返回 JSON：{{\"peers\": [\"600519\", \"000858\", ...]}}"
                    f"代码必须是标准的 6 位 A 股数字或美股字母代码。"
                    f"只返回 JSON，不要解释。"
                ),
            },
        ]

        try:
            response = await _call_model(
                prompt, self._task.get("model"), response_format="json"
            )
            data = json.loads(response) if isinstance(response, str) else response
            peer_tickers = data.get("peers", [])[:3] if isinstance(data, dict) else []

            if not peer_tickers:
                return PeerResult(success=False, source="llm",
                                  error="no_peers_generated")

            # 逐个验证 + 补查 PE/PB/ROE
            peers = []
            for pt in peer_tickers:
                peer_info = await self._fetch_peer_data(str(pt))
                if peer_info:
                    peers.append(peer_info)

            if not peers:
                return PeerResult(success=False, source="llm",
                                  error="all_peers_invalid")

            logger.info(
                f"[LLMPeerSource] {ticker} → LLM 生成同行: "
                f"{[p.ticker for p in peers]}"
            )
            return PeerResult(peers=peers, source="llm")

        except Exception as e:
            return PeerResult(success=False, source="llm", error=str(e))

    async def _fetch_peer_data(self, peer_ticker: str) -> Optional[PeerInfo]:
        """根据 ticker 格式路由到 FMP/雪球，补查 PE/PB/ROE。"""
        try:
            is_ashare = peer_ticker.isdigit() and len(peer_ticker) in (5, 6)
            if is_ashare:
                from .xueqiu_finance import XueqiuDataTool
                tool = XueqiuDataTool(peer_ticker)
            else:
                from .financial_data import FinancialDataTool
                tool = FinancialDataTool(peer_ticker)

            ov = await tool.get_stock_overview()
            if not ov or not ov.get("name"):
                return None
            return PeerInfo(
                ticker=peer_ticker,
                name=ov.get("name", ""),
                market_cap=ov.get("market_cap"),
                pe=ov.get("pe_ratio"),
                pb=ov.get("pb_ratio"),
                roe=ov.get("roe"),
                revenue_growth=ov.get("revenue_growth"),
            )
        except Exception as e:
            logger.debug(f"[LLMPeerSource] peer {peer_ticker} 验证失败: {e}")
            return None


# ======================================================================
# 编排层
# ======================================================================

class PeerResolver:
    """同行解析器：按优先级链依次尝试数据源，首个成功即返回。

    优先级链：
        FMP API（动态，全覆盖美股）→ YAML config（静态，零成本，含 A 股）→ LLM（兜底）

    每个 source 独立返回 PeerResult。resolver 负责：
    1. 按序调用
    2. 首个 success=True 且 peers 非空即返回
    3. 全部失败时返回空列表并记录覆盖盲区日志
    """

    def __init__(self, task: dict, fmp_fetcher=None,
                 peer_info_fetcher=None, config_path: str = None):
        """
        Args:
            task: 任务配置 dict（含 model 等），用于 LLM 兜底
            fmp_fetcher: 同步 FMP GET 函数，None 时跳过 FMP API
            peer_info_fetcher: 异步回调 (ticker:str) -> PeerInfo，
                               FMP API 拿到 ticker 后补查 PE/PB/ROE
            config_path: YAML 配置路径，None 用默认路径
        """
        self._sources: list[AbstractPeerSource] = []

        if fmp_fetcher is not None:
            self._breaker = CircuitBreaker()
            self._sources.append(FMPStockPeersSource(
                fmp_fetcher=fmp_fetcher,
                circuit_breaker=self._breaker,
                peer_info_fetcher=peer_info_fetcher,
            ))

        self._sources.append(YAMLIndustrySource(config_path=config_path))
        self._sources.append(LLMPeerSource(task=task))

    async def resolve(self, ticker: str, **context) -> list[PeerInfo]:
        """解析同行列表。

        Args:
            ticker: 目标股票代码
            **context: 上下文数据（overview dict / industry str 等）

        Returns:
            list[PeerInfo]，最多 6 个；全部失败时返回 []
        """
        source_chain = []
        for source in self._sources:
            try:
                result = await source.fetch(ticker, **context)
            except Exception as e:
                source_chain.append(f"{source.__class__.__name__}:error({e})")
                continue

            if result.success and result.peers:
                source_chain.append(f"{result.source}:hit({len(result.peers)})")
                logger.info(
                    f"[PeerResolver] {ticker} resolved via "
                    f"{' → '.join(source_chain)}"
                )
                return result.peers[:6]
            else:
                source_chain.append(
                    f"{result.source}:miss({result.error})"
                )

        logger.warning(
            f"[PEER_COVERAGE_GAP] ticker={ticker} "
            f"chain={' → '.join(source_chain)} peers=0"
        )
        return []
