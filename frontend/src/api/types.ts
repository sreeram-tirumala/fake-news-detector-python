export interface TopTerm {
  word: string;
  shap_value: number;
}

export interface PredictResponse {
  label: "real" | "fake";
  probability: number;
  top_terms: TopTerm[];
}

export interface ExplainResponse {
  rationale: string;
}

export interface UrlPredictResponse extends PredictResponse {
  fetched_title: string | null;
  fetched_text: string;
  fetch_method: "trafilatura" | "claude_web_fetch";
}

export interface CorroborationSource {
  url: string;
  title: string;
  stance: "supports" | "contradicts" | "unrelated";
  note: string;
}

export interface CorroborateResponse {
  verdict: "corroborated" | "contradicted" | "unverifiable" | "mixed";
  rationale: string;
  sources: CorroborationSource[];
  raw_search_queries: string[];
}

export interface ClassCounts {
  [label: string]: number;
}

export interface NullRates {
  [column: string]: number;
}

export interface TermCount {
  term: string;
  count: number;
}

export interface TopTermsData {
  real: TermCount[];
  fake: TermCount[];
}

export interface WeeklyCount {
  week: string;
  label: string;
  count: number;
}

export interface PcaPoint {
  x: number;
  y: number;
  label: string;
  cluster: number;
}

export interface PcaData {
  points: PcaPoint[];
}

export interface Topic {
  topic: number;
  terms: string[];
}

export interface ClassMetric {
  precision: number;
  recall: number;
  "f1-score": number;
  support: number;
}

export interface Metrics {
  report: Record<string, ClassMetric | number>;
  confusion_matrix: number[][];
  curves: {
    roc: { fpr: number[]; tpr: number[]; auc: number };
    pr: { precision: number[]; recall: number[] };
  };
}
