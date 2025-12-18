"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";

export function TopNavClient() {
  const [apiBase, setApiBase] = useState<string>("");

  useEffect(() => {
    setApiBase(
      process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000"
    );
  }, []);

  return (
    <div className="flex items-center gap-3">
      <Badge variant="outline">Local</Badge>
      <span className="text-xs text-gray-500">
        API: <span className="font-medium text-gray-900">{apiBase}</span>
      </span>
    </div>
  );
}

