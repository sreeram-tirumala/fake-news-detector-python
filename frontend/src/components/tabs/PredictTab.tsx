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
import { corroborate, explainPrediction, predict, predictFromUrl } from "../../api/client";
import type { CorroborateResponse, PredictResponse } from "../../api/types";

type InputMode = "text" | "url";

const VERDICT_COLORS: Record<CorroborateResponse["verdict"], string> = {
  corroborated: "#3182ce",
  contradicted: "#c53030",
  unverifiable: "#718096",
  mixed: "#805ad5",
};

export default function PredictTab() {
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [headline, setHeadline] = useState("");
  const [body, setBody] = useState("");
  const [url, setUrl] = useState("");

  const [explainWithClaude, setExplainWithClaude] = useState(false);
  const [factCheck, setFactCheck] = useState(false);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [fetchedTitle, setFetchedTitle] = useState<string | null>(null);
  const [fetchMethod, setFetchMethod] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  const [rationale, setRationale] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);

  const [corrobResult, setCorrobResult] = useState<CorroborateResponse | null>(null);
  const [corroborating, setCorroborating] = useState(false);
  const [corrobError, setCorrobError] = useState<string | null>(null);

  async function handlePredict() {
    setPredictError(null);
    setPrediction(null);
    setRationale(null);
    setExplainError(null);
    setCorrobResult(null);
    setCorrobError(null);
    setPredicting(true);

    let result: PredictResponse;
    let text: string;

    try {
      if (inputMode === "url") {
        const r = await predictFromUrl(url);
        result = r;
        text = r.fetched_text;
        setFetchedTitle(r.fetched_title);
        setFetchMethod(r.fetch_method);
      } else {
        text = `${headline} ${body}`.trim();
        if (!text) {
          setPredictError("Please paste some text.");
          setPredicting(false);
          return;
        }
        result = await predict(headline, body);
        setFetchedTitle(null);
        setFetchMethod(null);
      }
      setPrediction(result);
    } catch (e) {
      setPredictError(e instanceof Error ? e.message : String(e));
      setPredicting(false);
      return;
    }

    setPredicting(false);

    if (explainWithClaude && result.top_terms.length > 0) {
      setExplaining(true);
      explainPrediction(text, result.label, result.probability, result.top_terms)
        .then((r) => setRationale(r.rationale))
        .catch((e) => setExplainError(e instanceof Error ? e.message : String(e)))
        .finally(() => setExplaining(false));
    }

    if (factCheck) {
      setCorroborating(true);
      corroborate(text, inputMode === "url" ? url : undefined)
        .then(setCorrobResult)
        .catch((e) => setCorrobError(e instanceof Error ? e.message : String(e)))
        .finally(() => setCorroborating(false));
    }
  }

  const chartData = prediction?.top_terms
    .slice()
    .reverse()
    .map((t) => ({ word: t.word, value: t.shap_value })) ?? [];

  return (
    <div className="tab-panel">
      <div className="card">
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
            <input
              type="radio"
              checked={inputMode === "url"}
              onChange={() => setInputMode("url")}
            />
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
              rows={8}
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

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={explainWithClaude}
            onChange={(e) => setExplainWithClaude(e.target.checked)}
          />
          Explain with Claude Haiku (plain-English rationale from SHAP tokens)
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={factCheck}
            onChange={(e) => setFactCheck(e.target.checked)}
          />
          Fact-check with live web search (Claude Sonnet 5)
        </label>

        <button onClick={handlePredict} disabled={predicting} className="predict-btn">
          {predicting ? "Analyzing..." : "Predict"}
        </button>

        {predictError && <p className="error-text">{predictError}</p>}
      </div>

      {prediction && (
        <>
          {/* Tier 1: statistical model -- the sole classifier */}
          <div className="card tier-1">
            {fetchedTitle && (
              <p className="caption">
                Fetched via {fetchMethod}: <strong>{fetchedTitle}</strong>
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
            <p className="caption">
              This is the sole classification. Everything below is supplementary context, not a
              second opinion.
            </p>
          </div>

          {/* Tier 2: SHAP + Claude Haiku narration */}
          <div className="card tier-2">
            <h3>Why the model said this</h3>
            {chartData.length > 0 ? (
              <>
                <p className="caption">
                  Words with the largest SHAP contribution to this prediction. Positive pushes
                  toward REAL, negative pushes toward FAKE.
                </p>
                <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 32)}>
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
              </>
            ) : (
              <p className="caption">No clean SHAP tokens to display for this text.</p>
            )}

            {explainWithClaude && (
              <div className="rationale-box">
                <h4>Plain-English rationale (Claude Haiku)</h4>
                {explaining && <p className="caption">Asking Claude to narrate the SHAP evidence...</p>}
                {explainError && <p className="error-text">Claude explanation unavailable: {explainError}</p>}
                {rationale && <p>{rationale}</p>}
              </div>
            )}
          </div>

          {/* Tier 3: independent web corroboration -- never a real/fake verdict */}
          {factCheck && (
            <div className="card tier-3">
              <h3>Independent web corroboration</h3>
              {corroborating && (
                <p className="caption">Searching the web for corroborating or contradicting sources...</p>
              )}
              {corrobError && <p className="error-text">Corroboration unavailable: {corrobError}</p>}
              {corrobResult && (
                <>
                  <span
                    className="verdict-badge"
                    style={{ backgroundColor: VERDICT_COLORS[corrobResult.verdict] }}
                  >
                    {corrobResult.verdict}
                  </span>
                  <p>{corrobResult.rationale}</p>
                  <details>
                    <summary>Sources checked ({corrobResult.sources.length})</summary>
                    <ul>
                      {corrobResult.sources.map((s) => (
                        <li key={s.url}>
                          <strong>{s.stance}</strong> —{" "}
                          <a href={s.url} target="_blank" rel="noreferrer">
                            {s.title || s.url}
                          </a>
                          <div className="caption">{s.note}</div>
                        </li>
                      ))}
                    </ul>
                  </details>
                  <p className="caption">
                    This reflects what independent web sources say about the article's claims. It
                    is not a real/fake verdict and does not override the model's prediction above.
                  </p>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
