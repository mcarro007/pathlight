"use client";

import { useMemo, useState } from "react";
import { apiPost } from "@/lib/api";

type JobRow = Record<string, any>;

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div className="bg-white border rounded p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="font-medium">{label}</div>
        <div className="text-sm text-gray-600">{pct}%</div>
      </div>
      <div className="w-full h-2 bg-gray-100 rounded">
        <div className="h-2 rounded bg-gray-900" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function ConsumerPage() {
  const [skills, setSkills] = useState(
    "Python, data analysis, dashboards, stakeholder communication"
  );
  const [interests, setInterests] = useState("healthcare, public sector, research");
  const [experience, setExperience] = useState("Any");
  const [location, setLocation] = useState("");
  const [k, setK] = useState(15);

  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [jobsErr, setJobsErr] = useState<string | null>(null);

  const [jdText, setJdText] = useState("");
  const [loadingEval, setLoadingEval] = useState(false);
  const [evalErr, setEvalErr] = useState<string | null>(null);
  const [evalData, setEvalData] = useState<any>(null);

  const query = useMemo(() => {
    return `Skills: ${skills}\nInterests: ${interests}\nExperience: ${experience}`.trim();
  }, [skills, interests, experience]);

  async function runSearch() {
    setJobsErr(null);
    setLoadingJobs(true);
    try {
      const resp = await apiPost("/consumer/search", {
        query,
        k,
        location_filter: location,
      });
      setJobs(resp?.rows ?? resp ?? []);
    } catch (e: any) {
      setJobsErr(e?.message ?? "Search failed");
    } finally {
      setLoadingJobs(false);
    }
  }

  async function runEval() {
    setEvalErr(null);
    setLoadingEval(true);
    setEvalData(null);
    try {
      const resp = await apiPost("/consumer/evaluate-jd", { jd_text: jdText });
      setEvalData(resp);
    } catch (e: any) {
      setEvalErr(e?.message ?? "Evaluation failed");
    } finally {
      setLoadingEval(false);
    }
  }

  const cols = useMemo(() => {
    return ["match_score", "title", "company", "location", "salary_annual_clean", "_source_file"];
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-semibold">Consumer</h2>
        <p className="text-sm text-gray-600 mt-1">
          Skill-based job suggestions + job description screening (bias / realism / complexity).
        </p>
      </div>

      <div className="bg-white border rounded p-6">
        <h3 className="text-lg font-semibold mb-4">Suggest job postings to apply for</h3>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium">Your skillset</label>
            <textarea
              className="mt-1 w-full border rounded p-2 h-24"
              value={skills}
              onChange={(e) => setSkills(e.target.value)}
            />
          </div>

          <div>
            <label className="text-sm font-medium">Interests (optional)</label>
            <textarea
              className="mt-1 w-full border rounded p-2 h-24"
              value={interests}
              onChange={(e) => setInterests(e.target.value)}
            />
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="text-sm font-medium">Experience</label>
              <select
                className="mt-1 w-full border rounded p-2"
                value={experience}
                onChange={(e) => setExperience(e.target.value)}
              >
                <option>Any</option>
                <option>Entry</option>
                <option>Mid</option>
                <option>Senior</option>
              </select>
            </div>

            <div className="flex-1">
              <label className="text-sm font-medium">Location filter (optional)</label>
              <input
                className="mt-1 w-full border rounded p-2"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g., New York, Remote"
              />
            </div>

            <div className="w-28">
              <label className="text-sm font-medium">Top K</label>
              <input
                type="number"
                className="mt-1 w-full border rounded p-2"
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                min={5}
                max={50}
              />
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={runSearch}
            disabled={loadingJobs}
            className="px-4 py-2 rounded bg-gray-900 text-white disabled:opacity-60"
          >
            {loadingJobs ? "Searching..." : "Find matches"}
          </button>
          {jobsErr && <span className="text-sm text-red-600">{jobsErr}</span>}
        </div>

        {jobs.length > 0 && (
          <div className="mt-6">
            <div className="text-sm text-gray-600 mb-2">Showing {jobs.length} results</div>
            <div className="overflow-auto border rounded">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    {cols.map((c) => (
                      <th key={c} className="text-left p-2 font-medium">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((r, idx) => (
                    <tr key={idx} className="border-b last:border-b-0">
                      {cols.map((c) => (
                        <td key={c} className="p-2">
                          {String(r?.[c] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white border rounded p-6">
        <h3 className="text-lg font-semibold mb-2">Screen a job description before you apply</h3>

        <label className="text-sm font-medium">Job description</label>
        <textarea
          className="mt-1 w-full border rounded p-2 h-44"
          value={jdText}
          onChange={(e) => setJdText(e.target.value)}
          placeholder="Paste the job description here..."
        />

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={runEval}
            disabled={loadingEval || jdText.trim().length < 40}
            className="px-4 py-2 rounded bg-gray-900 text-white disabled:opacity-60"
          >
            {loadingEval ? "Analyzing..." : "Analyze JD"}
          </button>

          {evalErr && <span className="text-sm text-red-600">{evalErr}</span>}
        </div>

        {evalData && (
          <div className="mt-6 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ScoreBar label="Bias risk" value={evalData?.scores?.bias_risk ?? 0} />
              <ScoreBar
                label="Requirements intensity"
                value={evalData?.scores?.requirements_intensity ?? 0}
              />
              <ScoreBar label="Clarity / readability" value={evalData?.scores?.clarity ?? 0} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

