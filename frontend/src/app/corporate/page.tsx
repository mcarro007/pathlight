"use client";

import { useState } from "react";

export default function CorporatePage() {
  const [requirements, setRequirements] = useState("");
  const [result, setResult] = useState<string | null>(null);

  function generateJD() {
    // placeholder until backend hook
    setResult(
      "Generated Job Description will appear here.\n\n" +
      "Inputs:\n" +
      requirements
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Corporate – Job Builder</h1>

      <div className="bg-white border rounded-xl p-6 space-y-4">
        <label className="text-sm font-medium">
          What are you looking for in this role?
        </label>

        <textarea
          className="w-full h-40 border rounded p-3"
          placeholder="Describe responsibilities, skills, level, constraints…"
          value={requirements}
          onChange={(e) => setRequirements(e.target.value)}
        />

        <button
          onClick={generateJD}
          className="px-4 py-2 rounded bg-gray-900 text-white"
        >
          Generate Job Description
        </button>
      </div>

      {result && (
        <div className="bg-white border rounded-xl p-6 whitespace-pre-wrap">
          {result}
        </div>
      )}
    </div>
  );
}

