@'
import { ReactNode } from "react";

export function EmptyState({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="border rounded-2xl bg-background p-6 text-center space-y-2">
      <div className="font-medium">{title}</div>
      {subtitle ? <div className="text-sm text-muted-foreground">{subtitle}</div> : null}
      {action ? <div className="pt-2 flex justify-center">{action}</div> : null}
    </div>
  );
}
'@ | Set-Content -Encoding utf8 .\frontend\src\components\shared\empty-state.tsx
