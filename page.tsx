"use client";

import { useMemo, useState } from "react";
import {
  Activity, BarChart3, BriefcaseBusiness, BrainCircuit, ChevronDown,
  CircleHelp, Command, FileText, Globe2, LayoutDashboard, Newspaper,
  Search, Settings2, ShieldCheck, Wallet
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

type BacktestResponse = {
  weights: Record<string, number>;
  metrics: Record<string, number>;
  final_value: number;
  total_contributed: number;
  history: { date: string; portfolio_value: number; contributed: number }[];
  methodology: { limitations: string[] };
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nav = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Markets", icon: BarChart3 },
  { label: "Portfolios", icon: BriefcaseBusiness },
  { label: "Backtesting", icon: Activity },
  { label: "Macro research", icon: Globe2 },
  { label: "Market news", icon: Newspaper },
  { label: "AI analyst", icon: BrainCircuit },
  { label: "Reports", icon: FileText },
];

const money = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);

export default function Home() {
  const [weightsText, setWeightsText] = useState("VOO:50, QQQM:30, BND:20");
  const [startYear, setStartYear] = useState(2015);
  const [initial, setInitial] = useState(10000);
  const [contribution, setContribution] = useState(0);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const parsedWeights = useMemo(() => {
    const pairs = weightsText.split(",").map((part) => part.trim()).filter(Boolean);
    const out: Record<string, number> = {};
    for (const pair of pairs) {
      const [ticker, raw] = pair.split(":").map((v) => v.trim());
      const value = Number(raw);
      if (!ticker || !Number.isFinite(value) || value <= 0) return null;
      out[ticker.toUpperCase()] = value;
    }
    return Object.keys(out).length ? out : null;
  }, [weightsText]);

  async function runBacktest() {
    setError("");
    setResult(null);
    if (!parsedWeights) {
      setError("Enter holdings as VOO:50, QQQM:30, BND:20. Weights are relative and will be normalized by the engine.");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/backtest/custom-fund`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          weights: parsedWeights,
          start_year: startYear,
          initial_investment: initial,
          annual_contribution: contribution,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Backtest failed.");
      setResult(payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to reach the API. Is the backend running?");
    } finally {
      setBusy(false);
    }
  }

  const latest = result?.history.at(-1);
  const gain = latest ? latest.portfolio_value - latest.contributed : 0;

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">A</div><div><b>APEX</b><span>INVESTMENT INTELLIGENCE</span></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>{nav.map((item, i) => <button key={item.label} className={`nav-item ${i === 0 ? "active" : ""}`}><item.icon size={17}/><span>{item.label}</span>{item.label === "AI analyst" && <span className="new-tag">AI</span>}</button>)}</nav>
        <div className="sidebar-bottom">
          <button className="nav-item"><ShieldCheck size={17}/> Client access</button>
          <button className="nav-item"><Settings2 size={17}/> Settings</button>
          <div className="profile"><div className="avatar">JJ</div><div><b>Workspace</b><span>Development environment</span></div><ChevronDown size={15}/></div>
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <div className="crumb">Workspace <span>/</span> Overview</div>
          <div className="top-actions"><div className="search"><Search size={15}/><span>Search securities, research...</span><kbd>⌘ K</kbd></div><span className="market-status"><i/> DATA CONNECTION READY</span><button className="icon-button"><CircleHelp size={18}/></button></div>
        </header>

        <div className="content">
          <div className="page-heading">
            <div><div className="eyebrow">FRIDAY, NOVEMBER 6 · DEVELOPMENT PREVIEW</div><h1>Investment overview</h1><p>Your research, portfolios and quantitative tools in one workspace.</p></div>
            <div className="mode-pill"><span/> RESEARCH MODE</div>
          </div>

          <div className="notice"><Command size={16}/><span><b>Connected to your Python quant engine.</b> Backtests below call the existing <code>engine.run_custom_fund()</code> function.</span></div>

          <div className="section-title"><h2>Market snapshot</h2><span>Live market feeds are not configured in this starter</span></div>
          <div className="metric-grid">
            <div className="metric-card"><span>BENCHMARK</span><b>—</b><small>Connect a market data provider</small></div>
            <div className="metric-card"><span>PORTFOLIO VALUE</span><b>{result ? money(result.final_value) : "—"}</b><small>{result ? "Latest backtest result" : "Run a portfolio simulation"}</small></div>
            <div className="metric-card"><span>BACKTEST GAIN / LOSS</span><b className={gain >= 0 ? "positive" : "negative"}>{result ? money(gain) : "—"}</b><small>Compared with contributions</small></div>
            <div className="metric-card"><span>RISK PROFILE</span><b>—</b><small>Risk analytics not yet configured</small></div>
          </div>

          <div className="work-grid">
            <section className="panel chart-panel">
              <div className="panel-heading"><div><h2>Portfolio performance</h2><p>Historical value versus contributed capital</p></div><span className="subtle-chip">BACKTEST</span></div>
              {result ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={result.history}><defs><linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#55a6ff" stopOpacity={0.28}/><stop offset="100%" stopColor="#55a6ff" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#253244" vertical={false}/><XAxis dataKey="date" tick={{fill:"#8493a7",fontSize:11}} tickLine={false} axisLine={false} minTickGap={40}/><YAxis tick={{fill:"#8493a7",fontSize:11}} tickFormatter={(v) => `$${Math.round(v/1000)}k`} tickLine={false} axisLine={false}/><Tooltip contentStyle={{background:"#111c2a",border:"1px solid #2a3a4e",borderRadius:10,color:"#e8eef6"}} formatter={(v) => money(Number(v))}/><Area type="monotone" dataKey="portfolio_value" name="Portfolio" stroke="#55a6ff" strokeWidth={2} fill="url(#equityFill)"/><Area type="monotone" dataKey="contributed" name="Contributions" stroke="#91a0b3" strokeDasharray="5 5" fill="none"/></AreaChart></ResponsiveContainer></div> : <div className="empty-chart"><BarChart3 size={34}/><b>No simulation selected</b><span>Build a portfolio below and run a historical backtest.</span></div>}
            </section>

            <section className="panel analyst-panel">
              <div className="panel-heading"><div><h2><BrainCircuit size={18}/> AI research analyst</h2><p>Research assistant · planned integration</p></div><span className="soon-chip">COMING NEXT</span></div>
              <div className="analyst-prompt"><span>EXAMPLE RESEARCH PROMPT</span><p>Compare a 60/40 portfolio with a diversified equity portfolio during periods of rising inflation.</p></div>
              <div className="analyst-foot"><span>AI will use verified data and actual engine outputs.</span><button disabled>Configure AI</button></div>
            </section>
          </div>

          <section className="panel builder">
            <div className="panel-heading"><div><h2><Wallet size={18}/> Portfolio laboratory</h2><p>Run your existing historical custom-fund simulation through the new API.</p></div><span className="subtle-chip">CONNECTED TO PYTHON</span></div>
            <div className="form-grid">
              <label className="field wide"><span>HOLDINGS AND RELATIVE WEIGHTS</span><input value={weightsText} onChange={(e) => setWeightsText(e.target.value)} placeholder="VOO:50, QQQM:30, BND:20"/><small>Format: TICKER:WEIGHT, separated by commas.</small></label>
              <label className="field"><span>START YEAR</span><input type="number" min={2005} max={new Date().getFullYear()} value={startYear} onChange={(e) => setStartYear(Number(e.target.value))}/></label>
              <label className="field"><span>INITIAL INVESTMENT ($)</span><input type="number" min={1} value={initial} onChange={(e) => setInitial(Number(e.target.value))}/></label>
              <label className="field"><span>ANNUAL CONTRIBUTION ($)</span><input type="number" min={0} value={contribution} onChange={(e) => setContribution(Number(e.target.value))}/></label>
            </div>
            {error && <div className="error">{error}</div>}
            <div className="builder-actions"><span>Historical results are not forecasts. Data availability may vary by ticker.</span><button className="primary-button" onClick={runBacktest} disabled={busy}>{busy ? "Running simulation..." : "Run historical backtest"} <span>→</span></button></div>
            {result && <div className="results">
              <div className="result-head"><b>Simulation results</b><span>{result.history[0]?.date} — {latest?.date}</span></div>
              <div className="result-metrics">{Object.entries(result.metrics).map(([key, value]) => <div key={key}><span>{key}</span><b>{key.toLowerCase().includes("drawdown") || key === "CAGR" || key === "Volatility" ? `${(value * 100).toFixed(2)}%` : key.toLowerCase().includes("value") ? money(value) : value.toFixed(3)}</b></div>)}</div>
              <details><summary>Methodology and limitations</summary><ul>{result.methodology.limitations.map((x) => <li key={x}>{x}</li>)}</ul></details>
            </div>}
          </section>

          <footer>APEX INVESTMENT INTELLIGENCE <span>DEVELOPMENT BUILD · NOT FOR CLIENT INVESTMENT DECISIONS</span></footer>
        </div>
      </section>
    </main>
  );
}
