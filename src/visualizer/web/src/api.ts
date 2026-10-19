import type { Catalogue, Example, RunRequest, RunResult } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const { detail } = await response.json();
      if (typeof detail === "string") message = detail;
      else if (Array.isArray(detail))
        // FastAPI validation errors: [{ loc: ["body", "problem", ...], msg }]
        message = detail
          .map((d: { loc: (string | number)[]; msg: string }) => `${d.loc.slice(1).join(".")}: ${d.msg}`)
          .join("\n");
    } catch {
      /* keep the status line */
    }
    throw new Error(message);
  }
  return response.json();
}

export const getCatalogue = () => request<Catalogue>("/api/catalogue");
export const getExamples = () => request<Example[]>("/api/examples");
export const runSearch = (body: RunRequest) =>
  request<RunResult>("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
