"use client";

import { FormEvent, useState } from "react";

type Recommendation = {
  sell: string;
  buy: string;
  sell_id?: number;
  buy_id?: number;
  cost_change: number;
  projected_gain: number;
  sell_predicted_points: number;
  buy_predicted_points: number;
  buy_uncertainty: number;
  buy_prediction_low: number;
  buy_prediction_high: number;
  uncertainty_level: string;
  sell_expected_minutes: number;
  buy_expected_minutes: number;
  sell_fixture_difficulty: number;
  buy_fixture_difficulty: number;
  buy_rotation_probability: number;
  reason: string;
};

type RecommendationResponse = {
  team_id: number;
  gameweek: number;
  bank: number;
  model: "xgboost" | "heuristic";
  recommendations: Recommendation[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [teamId, setTeamId] = useState("");
  const [data, setData] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setData(null);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/recommendations/${teamId.trim()}`);
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? "The recommendation service returned an error.");
      }
      setData((await response.json()) as RecommendationResponse);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to reach the recommendation service.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <nav className="topbar">
        <div className="brand"><span className="brand-mark">F</span> FPL / FIELD NOTES</div>
        <div className="status-group">
          <span className="status"><span className="status-dot" /> LIVE DATA</span>
          {data && (
            <span className={`model-indicator ${data.model === "xgboost" ? "model-xgboost" : "model-heuristic"}`}>
              {data.model === "xgboost" ? "XGBoost model" : "Heuristic model"}
            </span>
          )}
        </div>
      </nav>
      <section className="hero">
        <p className="eyebrow">TRANSFER INTELLIGENCE / 01</p>
        <h1>Make your next move<br /><em>count.</em></h1>
        <p className="lede">A clear-eyed shortlist of the five transfers most likely to improve your squad over the weeks ahead.</p>
      </section>
      <section className="control-panel">
        <div><p className="panel-label">YOUR FPL TEAM</p><p className="panel-hint">Find the number in your team URL</p></div>
        <form onSubmit={handleSubmit} className="team-form">
          <label htmlFor="team-id" className="sr-only">FPL team ID</label>
          <input id="team-id" inputMode="numeric" pattern="[0-9]+" required value={teamId} onChange={(event) => setTeamId(event.target.value)} placeholder="e.g. 123456" />
          <button type="submit" disabled={loading}>{loading ? "SCOUTING..." : "FIND TRANSFERS  →"}</button>
        </form>
      </section>
      {error && <p className="error-message">{error} Check that the FastAPI server is running and the team ID is correct.</p>}
      {data && (
        <section className="results" aria-live="polite">
          <div className="results-heading">
            <div><p className="eyebrow">RECOMMENDATIONS / GW {data.gameweek}</p><h2>Your transfer shortlist</h2></div>
            <div className="bank">BANK <strong>£{data.bank.toFixed(1)}m</strong></div>
          </div>
          {data.recommendations.length === 0 ? <p className="empty">No positive-value transfers found for this squad right now.</p> : (
            <div className="recommendation-list">
              {data.recommendations.map((recommendation, index) => (
                <article className="recommendation" key={`${recommendation.sell_id ?? recommendation.sell}-${recommendation.buy_id ?? recommendation.buy}`}>
                  <span className="rank">0{index + 1}</span>
                  <div className="transfer"><span>SELL</span><strong>{recommendation.sell}</strong><span className="arrow">→</span><span>BUY</span><strong className="buy">{recommendation.buy}</strong></div>
                  <div className="gain"><strong>+{recommendation.projected_gain}</strong><span>PROJECTED PTS</span></div>
                  <div className="reason">
                    <strong>Main reason: {recommendation.reason}</strong>
                    <table className="comparison" aria-label={`Comparison of ${recommendation.sell} and ${recommendation.buy}`}>
                      <thead><tr><th>Metric</th><th>Sell</th><th>Buy</th></tr></thead>
                      <tbody>
                        <tr><th>Predicted points</th><td>{recommendation.sell_predicted_points.toFixed(2)}</td><td className="positive">{recommendation.buy_predicted_points.toFixed(2)}</td></tr>
                        <tr><th>Uncertainty range</th><td>—</td><td>{recommendation.buy_prediction_low.toFixed(2)}–{recommendation.buy_prediction_high.toFixed(2)}</td></tr>
                        <tr><th>Fixture difficulty</th><td>{recommendation.sell_fixture_difficulty.toFixed(1)}</td><td>{recommendation.buy_fixture_difficulty.toFixed(1)}</td></tr>
                        <tr><th>Expected minutes</th><td>{recommendation.sell_expected_minutes}</td><td>{recommendation.buy_expected_minutes}</td></tr>
                      </tbody>
                    </table>
                    <span className="metric-note">Lower fixture difficulty is better. Projected points cover the next five gameweeks.</span>
                    <span className={`uncertainty ${recommendation.uncertainty_level.toLowerCase()}`}><span className="uncertainty-dot" /> {recommendation.uncertainty_level} uncertainty · ±{recommendation.buy_uncertainty.toFixed(2)} points</span>
                  </div>
                  <span className={`cost ${recommendation.cost_change <= 0 ? "saving" : "spend"}`}>{recommendation.cost_change > 0 ? "+" : ""}£{recommendation.cost_change.toFixed(1)}m</span>
                </article>
              ))}
            </div>
          )}
        </section>
      )}
      {!data && !error && <p className="prompt">Enter your team ID to see the next five moves.</p>}
    </main>
  );
}
