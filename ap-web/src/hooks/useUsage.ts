import { useQuery } from "@tanstack/react-query";

import { authenticatedFetch } from "@/lib/identity";

export interface UsageWindow {
  used_percent: number | null;
  resets?: string | null;
  resets_at?: number | null;
  window_minutes?: number | null;
}

export interface ProviderUsage {
  available: boolean;
  provider?: string;
  plan_type?: string | null;
  session?: UsageWindow;
  week?: UsageWindow;
  error?: string;
}

export interface UsageData {
  claude: ProviderUsage;
  codex: ProviderUsage;
  fetched_at: number;
}

async function fetchUsage(): Promise<UsageData> {
  const res = await authenticatedFetch("/v1/usage");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as UsageData;
}

/** Polls the backend `/v1/usage` endpoint for live Claude + Codex limit usage. */
export function useUsage() {
  return useQuery({
    queryKey: ["usage"],
    queryFn: fetchUsage,
    refetchInterval: 60_000,
    staleTime: 45_000,
  });
}
