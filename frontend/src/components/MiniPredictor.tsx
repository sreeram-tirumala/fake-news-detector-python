import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { predict, predictFromUrl } from "../api/client";
import type { PredictResponse } from "../api/types";

type InputMode = "text" | "url";

/**
 * A trimmed-down version of PredictTab's tier-1 + tier-2: paste an article (or a
 * URL), get the model's label/probability and the SHAP chart behind it. Lets the
 * data-viz dashboard show a concrete article next to the aggregate charts, without
 * duplicating the full Predict tab's Claude narration/corroboration tiers.
 */
export default function MiniPredictor() {
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [headline, setHeadline] = useState("");
  const [body, setBody] = useState("");
  const [url, setUrl] = useState("");

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [fetchedTitle, setFetchedTitle] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict() {
    setError(null);
    setPrediction(null);
    setPredicting(true);
    try {
      if (inputMode === "url") {
        const r = await predictFromUrl(url);
        setPrediction(r);
        setFetchedTitle(r.fetched_title);
      } else {
        const text = `${headline} ${body}`.trim();
        if (!text) {
          setError("Please paste some text.");
          setPredicting(false);
          return;
        }
        setPrediction(await predict(headline, body));
        setFetchedTitle(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPredicting(false);
    }
  }

  const chartData =
    prediction?.top_terms
      .slice()
      .reverse()
      .map((t) => ({ word: t.word, value: t.shap_value })) ?? [];

  return (
    <div className="card mini-predictor">
      <h3>Try an article</h3>
      <p className="caption">
        Paste text or a URL to see the model's real/fake call and the SHAP chart behind it, in
        context with the dataset charts below. For the full explanation + web-corroboration
        experience, use the Predict tab.
      </p>

      <div className="input-mode-toggle">
        <label>
          <input
            type="radio"
            checked={inputMode === "text"}
            onChange={() => setInputMode("text")}
          />
          Paste text
        </label>
        <label>
          <input type="radio" checked={inputMode === "url"} onChange={() => setInputMode("url")} />
          Fetch from URL
        </label>
      </div>

      {inputMode === "text" ? (
        <>
          <input
            type="text"
            placeholder="Headline (optional)"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            className="text-input"
          />
          <textarea
            placeholder="Article text"
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={5}
            className="text-input"
          />
        </>
      ) : (
        <input
          type="text"
          placeholder="Article URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="text-input"
        />
      )}

      <button onClick={handlePredict} disabled={predicting} className="predict-btn">
        {predicting ? "Analyzing..." : "Predict"}
      </button>

      {error && <p className="error-text">{error}</p>}

      {prediction && (
        <div className="mini-predictor-result">
          {fetchedTitle && (
            <p className="caption">
              Fetched: <strong>{fetchedTitle}</strong>
            </p>
          )}
          <div className="metric-row">
            <div>
              <div className="metric-label">Predicted label</div>
              <div className="metric-value">{prediction.label}</div>
            </div>
            <div>
              <div className="metric-label">Probability (real)</div>
              <div className="metric-value">{prediction.probability.toFixed(3)}</div>
            </div>
          </div>

          {chartData.length > 0 && (
            <ResponsiveContainer width="100%" height={Math.max(180, chartData.length * 28)}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="word" width={100} />
                <Tooltip />
                <Bar dataKey="value">
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.value > 0 ? "#2a9d8f" : "#e76f51"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  );
}
