import type {
  ClassCounts,
  CorroborateResponse,
  ExplainResponse,
  Metrics,
  NullRates,
  PcaData,
  PredictResponse,
  TopTerm,
  TopTermsData,
  Topic,
  UrlPredictResponse,
  WeeklyCount,
} from "./types";

const BASE_URL = "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const getClassCounts = () => request<ClassCounts>("/data/class-counts");
export const getNullRates = () => request<NullRates>("/data/null-rates");
export const getTopTerms = () => request<TopTermsData>("/data/top-terms");
export const getWeeklyCounts = () => request<WeeklyCount[]>("/data/weekly-counts");
export const getPca = () => request<PcaData>("/data/pca");
export const getTopics = () => request<Topic[]>("/data/topics");
export const getMetrics = () => request<Metrics>("/data/metrics");

export const predict = (headline: string, body: string) =>
  request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ headline, body }),
  });

export const predictFromUrl = (url: string) =>
  request<UrlPredictResponse>("/predict/url", {
    method: "POST",
    body: JSON.stringify({ url }),
  });

export const explainPrediction = (
  text: string,
  label: "real" | "fake",
  probability: number,
  topTerms: TopTerm[],
) =>
  request<ExplainResponse>("/predict/explain", {
    method: "POST",
    body: JSON.stringify({ text, label, probability, top_terms: topTerms }),
  });

export const corroborate = (text: string, url?: string) =>
  request<CorroborateResponse>("/corroborate", {
    method: "POST",
    body: JSON.stringify({ text, url: url ?? null }),
  });
